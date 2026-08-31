package sysmonitor

import (
	"testing"
	"time"
)

// This is a  illustrative tegrastats line
const sampleLine = `RAM 3245/7620MB (lfb 12x4MB) SWAP 0/3810MB (cached 0MB) ` +
	`CPU [12%@1728,8%@1728] EMC_FREQ 12%@2133 GR3D_FREQ 45%@930 ` +
	`tj@45.5C soc0@43.5C soc2@44.0C ` +
	`VDD_IN 4521mW/4521mW VDD_CPU_GPU_CV 1234mW/1234mW VDD_SOC 987mW/987mW`

func TestParseTegrastatsLine_FullLine(t *testing.T) {
	f := parseTegrastatsLine(sampleLine)
	if !f.HasGPU || f.GPUUtilPct != 45 {
		t.Errorf("GPUUtilPct: got %v (has=%v), want 45", f.GPUUtilPct, f.HasGPU)
	}
	if !f.HasRAM || f.RAMUsedMB != 3245 || f.RAMTotalMB != 7620 {
		t.Errorf("RAM: got %d/%d (has=%v), want 3245/7620", f.RAMUsedMB, f.RAMTotalMB, f.HasRAM)
	}
	if !f.HasPower || f.PowerMW != 4521 {
		t.Errorf("PowerMW: got %v (has=%v), want 4521", f.PowerMW, f.HasPower)
	}
	if !f.HasTemps {
		t.Fatal("expected HasTemps=true")
	}
	if f.Temps["tj"] != 45.5 {
		t.Errorf("Temps[tj]: got %v, want 45.5", f.Temps["tj"])
	}
	if f.MaxTempC != 45.5 {
		t.Errorf("MaxTempC: got %v, want 45.5 (the highest of tj/soc0/soc2)", f.MaxTempC)
	}
}

func TestParseTegrastatsLine_EmptyLine(t *testing.T) {
	f := parseTegrastatsLine("")
	if f.HasGPU || f.HasRAM || f.HasPower || f.HasTemps {
		t.Errorf("expected no fields present for empty input, got %+v", f)
	}
}

func TestParseTegrastatsLine_PartialLine(t *testing.T) {
	// Only RAM and temps present, GPU/power should come back absent,
	// not zero
	line := "RAM 1000/8000MB tj@50.0C"
	f := parseTegrastatsLine(line)
	if !f.HasRAM || f.RAMUsedMB != 1000 {
		t.Errorf("expected RAM parsed, got %+v", f)
	}
	if !f.HasTemps || f.Temps["tj"] != 50.0 {
		t.Errorf("expected temps parsed, got %+v", f)
	}
	if f.HasGPU {
		t.Error("expected HasGPU=false when GR3D_FREQ absent from line")
	}
	if f.HasPower {
		t.Error("expected HasPower=false when VDD_IN absent from line")
	}
}

func TestParseTegrastatsLine_MaxTempPicksHighestSensor(t *testing.T) {
	line := "tj@30.0C soc0@55.5C soc2@40.0C"
	f := parseTegrastatsLine(line)
	if f.MaxTempC != 55.5 {
		t.Errorf("MaxTempC: got %v, want 55.5 (soc0, the highest reading)", f.MaxTempC)
	}
}

func TestParseNvpmodelOutput_ValidOutput(t *testing.T) {
	output := "NV Power Mode: MAXN\n0\n"
	mode, ok := parseNvpmodelOutput(output)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if mode != "MAXN" {
		t.Errorf("mode: got %q, want %q", mode, "MAXN")
	}
}

func TestParseNvpmodelOutput_MalformedOutput(t *testing.T) {
	_, ok := parseNvpmodelOutput("garbage, no power mode line here")
	if ok {
		t.Error("expected ok=false for output with no recognizable power mode line")
	}
}

// Monitor level test, confirm the atomic state carries forward fields
// a given line doesn't mention, rather than resetting them to zero
func TestMonitor_CarriesForwardFieldsAcrossLines(t *testing.T) {
	m := New()
	m.applyTegrastatsLine(sampleLine) // establishes GPUUtilPct=45, etc.
	firstState := m.Current()
	if !firstState.Valid || firstState.GPUUtilPct != 45 {
		t.Fatalf("setup failed, got %+v", firstState)
	}
	// A line that only reports RAM, GPU/power/temps should be UNCHANGED
	// from the previous state, not reset to zero
	m.applyTegrastatsLine("RAM 500/7620MB")
	secondState := m.Current()
	if secondState.RAMUsedMB != 500 {
		t.Errorf("expected RAM to update to 500, got %d", secondState.RAMUsedMB)
	}
	if secondState.GPUUtilPct != 45 {
		t.Errorf("expected GPUUtilPct to carry forward as 45, got %v -- "+
			"a line not mentioning GPU should not reset it", secondState.GPUUtilPct)
	}
	if secondState.PowerMW != 4521 {
		t.Errorf("expected PowerMW to carry forward as 4521, got %v", secondState.PowerMW)
	}
}

func TestMonitor_CurrentIsInvalidBeforeAnyReading(t *testing.T) {
	m := New()
	state := m.Current()
	if state.Valid {
		t.Error("expected Valid=false before any tegrastats line has been applied")
	}
}

func TestMonitor_TimestampAdvancesOnEachApply(t *testing.T) {
	m := New()
	m.applyTegrastatsLine(sampleLine)
	t1 := m.Current().Timestamp
	time.Sleep(2 * time.Millisecond) // ensure a measurable clock difference
	m.applyTegrastatsLine(sampleLine)
	t2 := m.Current().Timestamp
	if !t2.After(t1) {
		t.Errorf("expected timestamp to advance: t1=%v t2=%v", t1, t2)
	}
}
