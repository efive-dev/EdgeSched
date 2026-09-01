// Package workerpool implements a bounded worker pool for one inference
// engine: a fixed number of goroutines drawing from a bounded queue,
// calling into a Predictor (typically an engineclient.Client). Bounding
// both queue size and concurrency is the mechanism admission control is
// built on. Submit returning false is the "reject" signal an HTTP
// handler or routing policy can act on, rather than letting requests
// pile up indefinitely behind an already-saturated engine.
package workerpool

import (
	"context"
	"sync"

	pb "edgesched/scheduler/generated"
)

// Predictor is the subset of engineclient.Client this package depends on.
type Predictor interface {
	Predict(ctx context.Context, imageData []byte) (*pb.PredictResponse, error)
}

// Request is one unit of work submitted to a Pool. The caller owns
// Result and should read exactly one value from it after a successful
// Submit.
type Request struct {
	Ctx       context.Context
	ImageData []byte
	Result    chan<- Result
}

type Result struct {
	Response *pb.PredictResponse
	Err      error
}

// Pool bounds concurrency (a fixed number of worker goroutines) and
// queue depth (a bounded channel) for calls to one Predictor
type Pool struct {
	predictor Predictor
	queue     chan Request
	wg        sync.WaitGroup
}

// New creates a Pool and starts its worker goroutines immediately
func New(predictor Predictor, queueSize, concurrency int) *Pool {
	if concurrency < 1 {
		concurrency = 1
	}
	if queueSize < 0 {
		queueSize = 0
	}
	p := &Pool{
		predictor: predictor,
		queue:     make(chan Request, queueSize),
	}
	p.wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go p.worker()
	}
	return p
}

func (p *Pool) worker() {
	defer p.wg.Done()
	for req := range p.queue {
		resp, err := p.predictor.Predict(req.Ctx, req.ImageData)
		req.Result <- Result{Response: resp, Err: err}
	}
}

// Submit enqueues req. Returns false immediately, without blocking, if
// the queue is already full. The caller decides what "queue full" means:
// reject with an HTTP 503, try a different tier, etc
func (p *Pool) Submit(req Request) bool {
	select {
	case p.queue <- req:
		return true
	default:
		return false
	}
}

// QueueDepth returns how many requests are currently waiting in the
// queue (not counting ones already picked up by a worker)
func (p *Pool) QueueDepth() int {
	return len(p.queue)
}

// Close stops accepting new work and waits for all in-flight and
// already queued requests to finish.
func (p *Pool) Close() {
	close(p.queue)
	p.wg.Wait()
}
