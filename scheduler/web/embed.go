// Package web serves the live dashboard (Phase 5): a single static
// HTML/JS page
package web

import (
	"embed"
	"io/fs"
	"net/http"
)

//go:embed static
var staticFiles embed.FS

// Handler serves the dashboard's static files. Registered at "/" in
// main.go
func Handler() http.Handler {
	sub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		panic(err)
	}
	return http.FileServer(http.FS(sub))
}
