// Package sysmonitor polls Jetson system state (thermal, power, memory,
// GPU utilization) in the background and exposes the latest reading
package sysmonitor

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os/exec"
	"regexp"
	"strconv"
	"sync/atomic"
	"time"
)

// State is a snapshot of system conditions at one point in time.
type State struct {
	Timestamp  time.Time
	GPUUtilPct float64
	Temps      map[string]float64 // sensor name -> Celsius, e.g. "tj", "soc0"
	MaxTempC   float64            // convenience: highest reading across all sensors
	PowerMW    float64            // VDD_IN -- total system power draw
	RAMUsedMB  int
	RAMTotalMB int
	PowerMode  string // from `nvpmodel -q`, e.g. "MAXN"
	Valid      bool   // false until the first successful tegrastats parse
}

type Monitor struct {
	state         atomic.Pointer[State]
	tegraInterval time.Duration
	nvpInterval   time.Duration
}

func New() *Monitor {
	m := &Monitor{
		tegraInterval: 1 * time.Second,
		nvpInterval:   5 * time.Second,
	}
	m.state.Store(&State{}) // zero value, Valid=false, until first real reading
	return m
}

// Current returns the latest known system state
func (m *Monitor) Current() State {
	return *m.state.Load()
}

// Run starts the background polling loops (tegrastats + nvpmodel) and
// blocks until ctx is canceled or one of them fails unrecoverably.
func (m *Monitor) Run(ctx context.Context) error {
	errCh := make(chan error, 2)
	go func() { errCh <- m.runTegrastats(ctx) }()
	go func() { errCh <- m.runNvpmodel(ctx) }()
	err := <-errCh
	if ctx.Err() != nil {
		return nil // canceled deliberately, not a real error
	}
	return err
}

// tegrastats parsing
var (
	gpuUtilRe = regexp.MustCompile(`GR3D_FREQ (\d+)%`)
	ramRe     = regexp.MustCompile(`RAM (\d+)/(\d+)MB`)
	powerRe   = regexp.MustCompile(`VDD_IN (\d+)mW`)
	tempRe    = regexp.MustCompile(`(\w+)@(-?[\d.]+)C`)
)

// tegraFields is the pure parse result of one tegrastats line
type tegraFields struct {
	GPUUtilPct float64
	HasGPU     bool
	RAMUsedMB  int
	RAMTotalMB int
	HasRAM     bool
	PowerMW    float64
	HasPower   bool
	Temps      map[string]float64
	MaxTempC   float64
	HasTemps   bool
}

// parseTegrastatsLine has no I/O, no shared state, so it can be
// unit tested directly against known input strings.
func parseTegrastatsLine(line string) tegraFields {
	var f tegraFields
	if match := gpuUtilRe.FindStringSubmatch(line); match != nil {
		if v, err := strconv.ParseFloat(match[1], 64); err == nil {
			f.GPUUtilPct = v
			f.HasGPU = true
		}
	}
	if match := ramRe.FindStringSubmatch(line); match != nil {
		used, uErr := strconv.Atoi(match[1])
		total, tErr := strconv.Atoi(match[2])
		if uErr == nil && tErr == nil {
			f.RAMUsedMB = used
			f.RAMTotalMB = total
			f.HasRAM = true
		}
	}
	if match := powerRe.FindStringSubmatch(line); match != nil {
		if v, err := strconv.ParseFloat(match[1], 64); err == nil {
			f.PowerMW = v
			f.HasPower = true
		}
	}
	temps := make(map[string]float64)
	maxTemp := -1000.0
	for _, tm := range tempRe.FindAllStringSubmatch(line, -1) {
		val, err := strconv.ParseFloat(tm[2], 64)
		if err != nil {
			continue
		}
		temps[tm[1]] = val
		if val > maxTemp {
			maxTemp = val
		}
	}
	if len(temps) > 0 {
		f.Temps = temps
		f.MaxTempC = maxTemp
		f.HasTemps = true
	}
	return f
}

func (m *Monitor) runTegrastats(ctx context.Context) error {
	intervalMs := int(m.tegraInterval / time.Millisecond)
	cmd := exec.CommandContext(ctx, "tegrastats", "--interval", strconv.Itoa(intervalMs))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("tegrastats stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting tegrastats (is it on PATH?): %w", err)
	}
	scanner := bufio.NewScanner(stdout)
	if scanner == nil {
		return fmt.Errorf("could not open scanner")
	}
	for scanner.Scan() {
		m.applyTegrastatsLine(scanner.Text())
	}
	if err := cmd.Wait(); err != nil && ctx.Err() == nil {
		return fmt.Errorf("tegrastats exited unexpectedly: %w", err)
	}
	return ctx.Err()
}

// applyTegrastatsLine merges one line's parsed fields into the stored
// state
func (m *Monitor) applyTegrastatsLine(line string) {
	fields := parseTegrastatsLine(line)
	prev := m.Current()
	next := prev
	next.Timestamp = time.Now()
	next.Valid = true
	if fields.HasGPU {
		next.GPUUtilPct = fields.GPUUtilPct
	}
	if fields.HasRAM {
		next.RAMUsedMB = fields.RAMUsedMB
		next.RAMTotalMB = fields.RAMTotalMB
	}
	if fields.HasPower {
		next.PowerMW = fields.PowerMW
	}
	if fields.HasTemps {
		next.Temps = fields.Temps
		next.MaxTempC = fields.MaxTempC
	}
	m.state.Store(&next)
}

// nvpmodel polling (power mode should not actually change but...)
var nvpModeRe = regexp.MustCompile(`NV Power Mode:\s*(\S+)`)

func parseNvpmodelOutput(output string) (mode string, ok bool) {
	match := nvpModeRe.FindStringSubmatch(output)
	if match == nil {
		return "", false
	}
	return match[1], true
}

func (m *Monitor) runNvpmodel(ctx context.Context) error {
	ticker := time.NewTicker(m.nvpInterval)
	defer ticker.Stop()
	for {
		m.pollNvpmodel(ctx)
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
		}
	}
}

func (m *Monitor) pollNvpmodel(ctx context.Context) {
	cmd := exec.CommandContext(ctx, "nvpmodel", "-q")
	out, err := cmd.Output()
	if err != nil {
		log.Printf("sysmonitor: nvpmodel query failed: %v", err)
		return
	}
	mode, ok := parseNvpmodelOutput(string(out))
	if !ok {
		log.Printf("sysmonitor: could not parse nvpmodel output: %q", string(out))
		return
	}
	prev := m.Current()
	next := prev
	next.PowerMode = mode
	m.state.Store(&next)
}
