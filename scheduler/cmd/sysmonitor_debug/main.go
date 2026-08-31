// Command sysmonitor_debug runs the system monitor standalone and prints
// its parsed state every 2 seconds
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"time"

	"edgesched/scheduler/internal/sysmonitor"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	m := sysmonitor.New()
	go func() {
		if err := m.Run(ctx); err != nil {
			fmt.Println("monitor error:", err)
		}
	}()
	fmt.Println("Polling... press Ctrl+C to stop.")
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			state := m.Current()
			b, _ := json.MarshalIndent(state, "", "  ")
			fmt.Println(string(b))
		}
	}
}
