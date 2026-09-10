// Command loadgen sends every image in a directory to the scheduler's
// /predict endpoint, with configurable client side concurrency, and
// reports aggregate latency/throughput/success statistics
package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var imageExts = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".bmp": true,
}

// predictResponse mirrors api.predictResponseJSON, kept as a separate
// deliberately loose definition here (this tool doesn't share Go types
// with the scheduler package)
type predictResponse struct {
	Engine        string     `json:"engine"`
	AutoRouted    bool       `json:"auto_routed"`
	PreprocessMs  float64    `json:"preprocess_ms"`
	InferenceMs   float64    `json:"inference_ms"`
	PostprocessMs float64    `json:"postprocess_ms"`
	Detections    []struct{} `json:"detections"`
}

type result struct {
	Path        string
	StatusCode  int
	Err         error
	WallMs      float64
	TimestampMs int64 // completion time (unix ms)
	Resp        predictResponse
}

// summaryJSON is written to -summary-json for automated aggregation
// across many runs
type summaryJSON struct {
	Label         string         `json:"label,omitempty"`
	Concurrency   int            `json:"concurrency"`
	DurationSec   float64        `json:"duration_sec"`
	Total         int            `json:"total"`
	Succeeded     int            `json:"succeeded"`
	Failed        int            `json:"failed"`
	ThroughputRPS float64        `json:"throughput_rps"`
	LatencyP50Ms  float64        `json:"latency_p50_ms"`
	LatencyP95Ms  float64        `json:"latency_p95_ms"`
	LatencyP99Ms  float64        `json:"latency_p99_ms"`
	LatencyMaxMs  float64        `json:"latency_max_ms"`
	EngineCounts  map[string]int `json:"engine_counts"`
	StatusCounts  map[string]int `json:"status_counts"` // string keys: JSON object keys can't be ints
}

// findImages is pure I/O but no network/HTTP
func findImages(dir string) ([]string, error) {
	var paths []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if imageExts[strings.ToLower(filepath.Ext(path))] {
			paths = append(paths, path)
		}
		return nil
	})
	return paths, err
}

// buildPredictURL is pure
func buildPredictURL(baseURL, engine string, budgetMs int) string {
	url := strings.TrimRight(baseURL, "/") + "/predict"
	var params []string
	if engine != "" {
		params = append(params, "engine="+engine)
	}
	if budgetMs > 0 {
		params = append(params, "latency_budget_ms="+strconv.Itoa(budgetMs))
	}
	if len(params) > 0 {
		url += "?" + strings.Join(params, "&")
	}
	return url
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p / 100 * float64(len(sorted)-1))
	return sorted[idx]
}

func sendOne(client *http.Client, baseURL, path, engine string, budgetMs int) result {
	data, err := os.ReadFile(path)
	if err != nil {
		return result{Path: path, Err: fmt.Errorf("read file: %w", err), TimestampMs: time.Now().UnixMilli()}
	}
	url := buildPredictURL(baseURL, engine, budgetMs)
	start := time.Now()
	resp, err := client.Post(url, "application/octet-stream", bytes.NewReader(data))
	wallMs := float64(time.Since(start).Microseconds()) / 1000.0
	completedAt := time.Now().UnixMilli() // captured once, used for every return path below
	if err != nil {
		return result{Path: path, Err: err, WallMs: wallMs, TimestampMs: completedAt}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return result{
			Path: path, StatusCode: resp.StatusCode, WallMs: wallMs, TimestampMs: completedAt,
			Err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body))),
		}
	}
	var pr predictResponse
	if err := json.Unmarshal(body, &pr); err != nil {
		return result{
			Path: path, StatusCode: resp.StatusCode, WallMs: wallMs, TimestampMs: completedAt,
			Err: fmt.Errorf("decode response: %w", err),
		}
	}
	return result{Path: path, StatusCode: resp.StatusCode, WallMs: wallMs, TimestampMs: completedAt, Resp: pr}
}

func writeCSV(path string, results []result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{
		"path", "timestamp_ms", "status_code", "error", "wall_ms", "engine", "auto_routed",
		"preprocess_ms", "inference_ms", "postprocess_ms", "num_detections",
	})
	for _, r := range results {
		errStr := ""
		if r.Err != nil {
			errStr = r.Err.Error()
		}
		w.Write([]string{
			r.Path,
			strconv.FormatInt(r.TimestampMs, 10),
			strconv.Itoa(r.StatusCode),
			errStr,
			fmt.Sprintf("%.2f", r.WallMs),
			r.Resp.Engine,
			strconv.FormatBool(r.Resp.AutoRouted),
			fmt.Sprintf("%.2f", r.Resp.PreprocessMs),
			fmt.Sprintf("%.2f", r.Resp.InferenceMs),
			fmt.Sprintf("%.2f", r.Resp.PostprocessMs),
			strconv.Itoa(len(r.Resp.Detections)),
		})
	}
	return nil
}

func main() {
	addr := flag.String("addr", "http://localhost:8080", "scheduler HTTP address")
	dir := flag.String("dir", "", "directory of images to send (required)")
	engine := flag.String("engine", "", "explicit engine override (empty = auto-routed)")
	budgetMs := flag.Int("latency-budget-ms", 0, "optional per-request latency budget (0 = server default)")
	concurrency := flag.Int("concurrency", 4, "number of concurrent in-flight requests")
	timeoutSec := flag.Int("timeout-sec", 60, "HTTP client timeout per request, in seconds")
	outCSV := flag.String("out", "", "optional path to write per-request results as CSV")
	summaryJSONPath := flag.String("summary-json", "", "optional path to write a machine-readable summary JSON")
	label := flag.String("label", "", "optional label included in the summary JSON (e.g. a policy name or concurrency level)")
	duration := flag.Duration("duration", 0,
		"if >0, cycle through -dir repeatedly for this long instead of sending each image once "+
			"(e.g. -duration=5m). Sustained mode, for steady-state benchmarking.")
	warmup := flag.Int("warmup", 0,
		"send this many requests first, sequentially, discarded from stats -- skips past cold-start effects")
	flag.Parse()
	if *dir == "" {
		log.Fatal("must specify -dir")
	}
	paths, err := findImages(*dir)
	if err != nil {
		log.Fatalf("failed to walk %s: %v", *dir, err)
	}
	if len(paths) == 0 {
		log.Fatalf("no images found in %s (looked for .jpg/.jpeg/.png/.bmp)", *dir)
	}
	client := &http.Client{Timeout: time.Duration(*timeoutSec) * time.Second}
	if *warmup > 0 {
		fmt.Printf("Warming up with %d requests (discarded from stats)...\n", *warmup)
		for i := 0; i < *warmup; i++ {
			sendOne(client, *addr, paths[i%len(paths)], *engine, *budgetMs)
		}
	}
	mode := "one-pass"
	if *duration > 0 {
		mode = "sustained, " + duration.String()
	}
	fmt.Printf("Found %d images in %s\n", len(paths), *dir)
	fmt.Printf("Mode: %s | concurrency=%d | addr=%s | engine=%q | budget=%dms\n\n",
		mode, *concurrency, *addr, *engine, *budgetMs)
	pathsCh := make(chan string)
	resultsCh := make(chan result)
	var wg sync.WaitGroup
	for i := 0; i < *concurrency; i++ {
		wg.Go(func() {
			for p := range pathsCh {
				resultsCh <- sendOne(client, *addr, p, *engine, *budgetMs)
			}
		})
	}
	overallStart := time.Now()
	go func() {
		defer close(pathsCh)
		if *duration > 0 {
			deadline := overallStart.Add(*duration)
			idx := 0
			for time.Now().Before(deadline) {
				pathsCh <- paths[idx%len(paths)]
				idx++
			}
		} else {
			for _, p := range paths {
				pathsCh <- p
			}
		}
	}()
	go func() {
		wg.Wait()
		close(resultsCh)
	}()
	var results []result
	completed := 0
	for r := range resultsCh {
		results = append(results, r)
		completed++
		if completed%50 == 0 {
			fmt.Printf("\r%d completed (%.0fs elapsed)", completed, time.Since(overallStart).Seconds())
		}
	}
	fmt.Println()
	overallElapsed := time.Since(overallStart)
	var wallTimes []float64
	successCount := 0
	failCount := 0
	engineCounts := map[string]int{}
	statusCounts := map[int]int{}
	for _, r := range results {
		if r.Err != nil {
			failCount++
			statusCounts[r.StatusCode]++
			continue
		}
		successCount++
		wallTimes = append(wallTimes, r.WallMs)
		engineCounts[r.Resp.Engine]++
	}
	sort.Float64s(wallTimes)
	throughput := float64(len(results)) / overallElapsed.Seconds()
	fmt.Println("\n=== Summary ===")
	fmt.Printf("Total:      %d\n", len(results))
	fmt.Printf("Succeeded:  %d\n", successCount)
	fmt.Printf("Failed:     %d\n", failCount)
	fmt.Printf("Wall time:  %.1fs\n", overallElapsed.Seconds())
	fmt.Printf("Throughput: %.2f req/sec\n", throughput)
	var p50, p95, p99, maxMs float64
	if len(wallTimes) > 0 {
		p50 = percentile(wallTimes, 50)
		p95 = percentile(wallTimes, 95)
		p99 = percentile(wallTimes, 99)
		maxMs = wallTimes[len(wallTimes)-1]
		fmt.Println("\nClient-observed round-trip latency (ms):")
		fmt.Printf("  p50: %.1f\n", p50)
		fmt.Printf("  p95: %.1f\n", p95)
		fmt.Printf("  p99: %.1f\n", p99)
		fmt.Printf("  max: %.1f\n", maxMs)
	}
	if len(engineCounts) > 0 {
		fmt.Println("\nRequests by engine:")
		names := make([]string, 0, len(engineCounts))
		for name := range engineCounts {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			fmt.Printf("  %s: %d\n", name, engineCounts[name])
		}
	}
	if failCount > 0 {
		fmt.Println("\nFailures by status code:")
		codes := make([]int, 0, len(statusCounts))
		for code := range statusCounts {
			codes = append(codes, code)
		}
		sort.Ints(codes)
		for _, code := range codes {
			fmt.Printf("  %d: %d\n", code, statusCounts[code])
		}
	}
	if *outCSV != "" {
		if err := writeCSV(*outCSV, results); err != nil {
			log.Printf("failed to write CSV: %v", err)
		} else {
			fmt.Printf("\nPer-request results written to %s\n", *outCSV)
		}
	}
	if *summaryJSONPath != "" {
		statusCountsStr := make(map[string]int, len(statusCounts))
		for code, n := range statusCounts {
			statusCountsStr[strconv.Itoa(code)] = n
		}
		summary := summaryJSON{
			Label:         *label,
			Concurrency:   *concurrency,
			DurationSec:   overallElapsed.Seconds(),
			Total:         len(results),
			Succeeded:     successCount,
			Failed:        failCount,
			ThroughputRPS: throughput,
			LatencyP50Ms:  p50,
			LatencyP95Ms:  p95,
			LatencyP99Ms:  p99,
			LatencyMaxMs:  maxMs,
			EngineCounts:  engineCounts,
			StatusCounts:  statusCountsStr,
		}
		f, err := os.Create(*summaryJSONPath)
		if err != nil {
			log.Printf("failed to create %s: %v", *summaryJSONPath, err)
		} else {
			defer f.Close()
			enc := json.NewEncoder(f)
			enc.SetIndent("", "  ")
			if err := enc.Encode(summary); err != nil {
				log.Printf("failed to write summary JSON: %v", err)
			} else {
				fmt.Printf("Summary JSON written to %s\n", *summaryJSONPath)
			}
		}
	}
}
