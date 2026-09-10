// Command statscraper polls the scheduler's /status endpoint at a fixed
// interval and writes a timestamped CSV of system state and per-engine
// queue depth over time.
// Meant to run CONCURRENTLY with a loadgen sustained run (-duration on
// loadgen) so the two CSVs can be joined on time for benchmark charts
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"syscall"
	"time"
)

type engineStatus struct {
	Name       string `json:"name"`
	QueueDepth int    `json:"queue_depth"`
}

type systemStatus struct {
	Valid      bool    `json:"valid"`
	MaxTempC   float64 `json:"max_temp_c"`
	PowerMW    float64 `json:"power_mw"`
	GPUUtilPct float64 `json:"gpu_util_pct"`
	RAMUsedMB  int     `json:"ram_used_mb"`
}

type statusResp struct {
	Policy  string         `json:"policy"`
	Engines []engineStatus `json:"engines"`
	System  systemStatus   `json:"system"`
}

func main() {
	addr := flag.String("addr", "http://localhost:8080", "scheduler HTTP address")
	interval := flag.Duration("interval", 1*time.Second, "poll interval")
	duration := flag.Duration("duration", 0, "how long to run (0 = until Ctrl+C/SIGTERM)")
	out := flag.String("out", "timeseries.csv", "output CSV path")
	flag.Parse()
	f, err := os.Create(*out)
	if err != nil {
		log.Fatalf("failed to create %s: %v", *out, err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	client := &http.Client{Timeout: 3 * time.Second}
	start := time.Now()
	// Engine columns are fixed after the first successful poll
	var engineCols []string
	headerWritten := false
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()
	fmt.Printf("Polling %s/status every %s -> %s\n", *addr, *interval, *out)
	if *duration > 0 {
		fmt.Printf("Will stop automatically after %s\n", *duration)
	} else {
		fmt.Println("Running until Ctrl+C (no -duration given)")
	}
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nStopped (signal received).")
			return
		case <-ticker.C:
			if *duration > 0 && time.Since(start) > *duration {
				fmt.Println("Duration elapsed, stopping.")
				return
			}
			resp, err := client.Get(*addr + "/status")
			if err != nil {
				log.Printf("poll failed: %v", err)
				continue
			}
			var st statusResp
			decodeErr := json.NewDecoder(resp.Body).Decode(&st)
			resp.Body.Close()
			if decodeErr != nil {
				log.Printf("decode failed: %v", decodeErr)
				continue
			}
			if !headerWritten {
				names := make([]string, 0, len(st.Engines))
				for _, e := range st.Engines {
					names = append(names, e.Name)
				}
				sort.Strings(names)
				engineCols = names
				header := []string{
					"timestamp_ms", "elapsed_sec", "policy", "valid",
					"max_temp_c", "power_mw", "gpu_util_pct", "ram_used_mb",
				}
				for _, n := range engineCols {
					header = append(header, "queue_"+n)
				}
				if err := w.Write(header); err != nil {
					log.Printf("failed to write header: %v", err)
				}
				headerWritten = true
			}
			queueByName := make(map[string]int, len(st.Engines))
			for _, e := range st.Engines {
				queueByName[e.Name] = e.QueueDepth
			}
			row := []string{
				strconv.FormatInt(time.Now().UnixMilli(), 10),
				fmt.Sprintf("%.1f", time.Since(start).Seconds()),
				st.Policy,
				strconv.FormatBool(st.System.Valid),
				fmt.Sprintf("%.3f", st.System.MaxTempC),
				fmt.Sprintf("%.1f", st.System.PowerMW),
				fmt.Sprintf("%.1f", st.System.GPUUtilPct),
				strconv.Itoa(st.System.RAMUsedMB),
			}
			for _, n := range engineCols {
				row = append(row, strconv.Itoa(queueByName[n]))
			}
			if err := w.Write(row); err != nil {
				log.Printf("failed to write row: %v", err)
			}
			w.Flush()
		}
	}
}
