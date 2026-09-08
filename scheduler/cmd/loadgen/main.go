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
	Path       string
	StatusCode int
	Err        error
	WallMs     float64
	Resp       predictResponse
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
		return result{Path: path, Err: fmt.Errorf("read file: %w", err)}
	}
	url := buildPredictURL(baseURL, engine, budgetMs)
	start := time.Now()
	resp, err := client.Post(url, "application/octet-stream", bytes.NewReader(data))
	wallMs := float64(time.Since(start).Microseconds()) / 1000.0
	if err != nil {
		return result{Path: path, Err: err, WallMs: wallMs}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return result{
			Path: path, StatusCode: resp.StatusCode, WallMs: wallMs,
			Err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body))),
		}
	}
	var pr predictResponse
	if err := json.Unmarshal(body, &pr); err != nil {
		return result{
			Path: path, StatusCode: resp.StatusCode, WallMs: wallMs,
			Err: fmt.Errorf("decode response: %w", err),
		}
	}
	return result{Path: path, StatusCode: resp.StatusCode, WallMs: wallMs, Resp: pr}
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
		"path", "status_code", "error", "wall_ms", "engine", "auto_routed",
		"preprocess_ms", "inference_ms", "postprocess_ms", "num_detections",
	})
	for _, r := range results {
		errStr := ""
		if r.Err != nil {
			errStr = r.Err.Error()
		}
		w.Write([]string{
			r.Path,
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
	fmt.Printf("Found %d images in %s\n", len(paths), *dir)
	fmt.Printf("Sending with concurrency=%d to %s (engine=%q, budget=%dms)\n\n",
		*concurrency, *addr, *engine, *budgetMs)
	client := &http.Client{Timeout: time.Duration(*timeoutSec) * time.Second}
	sem := make(chan struct{}, *concurrency)
	resultsCh := make(chan result, len(paths))
	var wg sync.WaitGroup
	overallStart := time.Now()
	for _, p := range paths {
		wg.Add(1)
		sem <- struct{}{}
		go func(path string) {
			defer wg.Done()
			defer func() { <-sem }()
			resultsCh <- sendOne(client, *addr, path, *engine, *budgetMs)
		}(p)
	}
	go func() {
		wg.Wait()
		close(resultsCh)
	}()
	var results []result
	completed := 0
	for r := range resultsCh {
		results = append(results, r)
		completed++
		if completed%50 == 0 || completed == len(paths) {
			fmt.Printf("\r%d/%d completed", completed, len(paths))
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
	fmt.Println("\n=== Summary ===")
	fmt.Printf("Total:      %d\n", len(results))
	fmt.Printf("Succeeded:  %d\n", successCount)
	fmt.Printf("Failed:     %d\n", failCount)
	fmt.Printf("Wall time:  %.1fs\n", overallElapsed.Seconds())
	fmt.Printf("Throughput: %.2f images/sec\n", float64(len(results))/overallElapsed.Seconds())
	if len(wallTimes) > 0 {
		fmt.Println("\nClient-observed round-trip latency (ms):")
		fmt.Printf("  p50: %.1f\n", percentile(wallTimes, 50))
		fmt.Printf("  p95: %.1f\n", percentile(wallTimes, 95))
		fmt.Printf("  p99: %.1f\n", percentile(wallTimes, 99))
		fmt.Printf("  max: %.1f\n", wallTimes[len(wallTimes)-1])
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
}
