// Package engineclient wraps a gRPC connection to a single inference
// engine process.
package engineclient

import (
	"context"
	"fmt"

	pb "edgesched/scheduler/generated"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	name string
	addr string
	conn *grpc.ClientConn
	stub pb.InferenceClient
}

// New connects to one inference engine at address addr.
// It does not block or dial immediatly
func New(name, addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("connecting to engine %q at %s: %w", name, addr, err)
	}
	return &Client{
		name: name,
		addr: addr,
		conn: conn,
		stub: pb.NewInferenceClient(conn),
	}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}

func (c *Client) Name() string { return c.name }
func (c *Client) Addr() string { return c.addr }

// HealthCheck asks the engine whethere it's serving
func (c *Client) HealthCheck(ctx context.Context) (*pb.HealthCheckResponse, error) {
	resp, err := c.stub.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err != nil {
		return nil, fmt.Errorf("health check on engine %q: %w", c.name, err)
	}
	return resp, nil
}

// Predict sends image bytes to the engine and returns its response
// The caller can control timeout and cancellation via ctx
func (c *Client) Predict(ctx context.Context, imageData []byte) (*pb.PredictResponse, error) {
	resp, err := c.stub.Predict(ctx, &pb.PredictRequest{ImageData: imageData})
	if err != nil {
		return nil, fmt.Errorf("predict on engine %q: %w", c.name, err)
	}
	return resp, nil
}
