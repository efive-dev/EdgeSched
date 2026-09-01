package workerpool

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	pb "edgesched/scheduler/generated"
)

type fakePredictor struct {
	delay       func() time.Duration // optional; simulates work taking time
	err         error                // optional; if set, every call returns this error
	mu          sync.Mutex
	inFlight    int
	maxInFlight int
	calls       int
}

func (f *fakePredictor) Predict(ctx context.Context, imageData []byte) (*pb.PredictResponse, error) {
	f.mu.Lock()
	f.inFlight++
	f.calls++
	if f.inFlight > f.maxInFlight {
		f.maxInFlight = f.inFlight
	}
	f.mu.Unlock()
	if f.delay != nil {
		time.Sleep(f.delay())
	}
	f.mu.Lock()
	f.inFlight--
	f.mu.Unlock()
	if f.err != nil {
		return nil, f.err
	}
	return &pb.PredictResponse{}, nil
}

func (f *fakePredictor) MaxInFlight() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.maxInFlight
}

func (f *fakePredictor) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func TestPool_RespectsConcurrencyBound(t *testing.T) {
	fp := &fakePredictor{delay: func() time.Duration { return 20 * time.Millisecond }}
	pool := New(fp, 50, 3) // concurrency=3
	defer pool.Close()
	const n = 20
	results := make(chan Result, n)
	for i := range n {
		if !pool.Submit(Request{Ctx: context.Background(), Result: results}) {
			t.Fatalf("Submit unexpectedly rejected at i=%d", i)
		}
	}
	for range n {
		<-results
	}
	if fp.MaxInFlight() > 3 {
		t.Errorf("observed maxInFlight=%d, want <= 3 (the configured concurrency)",
			fp.MaxInFlight())
	}
	if fp.Calls() != n {
		t.Errorf("expected %d calls, got %d", n, fp.Calls())
	}
}

func TestPool_SubmitReturnsFalseWhenQueueFull(t *testing.T) {
	block := make(chan struct{})
	fp := &fakePredictor{delay: func() time.Duration {
		<-block
		return 0
	}}
	pool := New(fp, 1, 1)
	defer func() {
		close(block)
		pool.Close()
	}()
	results := make(chan Result, 10)
	if !pool.Submit(Request{Ctx: context.Background(), Result: results}) {
		t.Fatal("first Submit should succeed")
	}
	time.Sleep(20 * time.Millisecond) // let the worker actually dequeue it
	if !pool.Submit(Request{Ctx: context.Background(), Result: results}) {
		t.Fatal("second Submit should succeed (queue has room for 1)")
	}
	done := make(chan bool, 1)
	go func() {
		done <- pool.Submit(Request{Ctx: context.Background(), Result: results})
	}()
	select {
	case ok := <-done:
		if ok {
			t.Error("expected third Submit to return false (queue full), got true")
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Submit blocked instead of returning false immediately")
	}
}

func TestPool_QueueDepthReflectsWaitingRequests(t *testing.T) {
	block := make(chan struct{})
	fp := &fakePredictor{delay: func() time.Duration {
		<-block
		return 0
	}}
	pool := New(fp, 5, 1) // concurrency=1
	defer func() {
		close(block)
		pool.Close()
	}()
	results := make(chan Result, 10)
	pool.Submit(Request{Ctx: context.Background(), Result: results}) // picked up immediately
	time.Sleep(20 * time.Millisecond)
	pool.Submit(Request{Ctx: context.Background(), Result: results}) // waits in queue
	pool.Submit(Request{Ctx: context.Background(), Result: results}) // waits in queue
	time.Sleep(20 * time.Millisecond)
	if depth := pool.QueueDepth(); depth != 2 {
		t.Errorf("QueueDepth() = %d, want 2 (one request is in-flight, not queued)", depth)
	}
}

func TestPool_ReturnsSuccessfulResponseViaResult(t *testing.T) {
	fp := &fakePredictor{}
	pool := New(fp, 5, 1)
	defer pool.Close()
	results := make(chan Result, 1)
	pool.Submit(Request{Ctx: context.Background(), Result: results})
	res := <-results
	if res.Err != nil {
		t.Errorf("expected nil error, got %v", res.Err)
	}
	if res.Response == nil {
		t.Error("expected non-nil response")
	}
}

func TestPool_PropagatesPredictorErrors(t *testing.T) {
	wantErr := errors.New("engine unavailable")
	fp := &fakePredictor{err: wantErr}
	pool := New(fp, 5, 1)
	defer pool.Close()
	results := make(chan Result, 1)
	pool.Submit(Request{Ctx: context.Background(), Result: results})
	res := <-results
	if !errors.Is(res.Err, wantErr) {
		t.Errorf("expected error %v, got %v", wantErr, res.Err)
	}
	if res.Response != nil {
		t.Errorf("expected nil response on error, got %+v", res.Response)
	}
}

func TestPool_CloseDrainsInFlightAndQueuedWork(t *testing.T) {
	fp := &fakePredictor{}
	pool := New(fp, 5, 2)
	results := make(chan Result, 3)
	for i := range 3 {
		if !pool.Submit(Request{Ctx: context.Background(), Result: results}) {
			t.Fatalf("Submit %d unexpectedly rejected", i)
		}
	}
	pool.Close() // must not return until all 3 have been processed
	for i := range 3 {
		select {
		case <-results:
		default:
			t.Errorf("expected all 3 results available after Close returns (missing result %d)", i)
		}
	}
}
