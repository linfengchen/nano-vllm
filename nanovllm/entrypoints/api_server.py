"""
OpenAI-compatible API server for nanovllm.
Simplified version based on vLLM's implementation.
"""
import asyncio
import json
from typing import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.entrypoints.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    DeltaMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    CompletionStreamResponseChoice,
    ModelList,
    ModelCard,
    UsageInfo,
    ErrorResponse,
)


class OpenAIServer:
    """OpenAI-compatible API server for nanovllm."""
    
    def __init__(self, model_path: str, **engine_kwargs):
        """
        Initialize the server.
        
        Args:
            model_path: Path to the model
            **engine_kwargs: Additional arguments for LLM engine
        """
        self.model_path = model_path
        self.engine_kwargs = engine_kwargs
        self.engine = None
        # Extract model name from path, handle trailing slashes
        model_name = model_path.rstrip("/").split("/")[-1]
        self.model_name = model_name if model_name else "model"
        self._init_engine()
        
    def _init_engine(self):
        """Initialize the LLM engine."""
        if self.engine is None:
            print(f"Initializing nanovllm engine with model: {self.model_path}")
            self.engine = LLM(self.model_path, **self.engine_kwargs)
            print("Engine initialized successfully")
    
    def _create_sampling_params(
        self,
        temperature: float = 1.0,
        max_tokens: int = 64,
        ignore_eos: bool = False,
    ) -> SamplingParams:
        """Create sampling parameters from request."""
        return SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
    
    def _format_chat_prompt(self, messages: list) -> str:
        """Format chat messages into a prompt string."""
        # Simple format: concatenate messages with role prefixes
        prompt_parts = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add assistant prefix for generation
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> UsageInfo:
        """Calculate token usage."""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
    async def handle_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Handle chat completion request (non-streaming)."""
        self._init_engine()
        
        # Format prompt from messages
        prompt = self._format_chat_prompt(request.messages)
        
        # Create sampling parameters
        sampling_params = self._create_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        
        # Generate
        outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
        output = outputs[0]
        
        # Calculate tokens (approximate)
        prompt_tokens = len(self.engine.tokenizer.encode(prompt))
        completion_tokens = len(output["token_ids"])
        
        # Create response
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=output["text"]),
            finish_reason="stop",
        )
        
        response = ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=self._calculate_usage(prompt_tokens, completion_tokens),
        )
        
        return response
    
    async def handle_chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Handle streaming chat completion request."""
        self._init_engine()
        
        # Format prompt from messages
        prompt = self._format_chat_prompt(request.messages)
        
        # For streaming, we'll simulate by generating first then streaming
        # In a real implementation, you'd integrate with a streaming generation API
        sampling_params = self._create_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        
        # Generate (this is a simplified streaming simulation)
        outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
        output_text = outputs[0]["text"]
        
        # Stream response character by character (simplified)
        # First chunk with role
        first_chunk = ChatCompletionStreamResponse(
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"
        
        # Content chunks (send every 10 characters for smoother streaming)
        chunk_size = 10
        for i in range(0, len(output_text), chunk_size):
            chunk_text = output_text[i:i + chunk_size]
            chunk = ChatCompletionStreamResponse(
                model=request.model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content=chunk_text),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        # Final chunk
        final_chunk = ChatCompletionStreamResponse(
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    
    async def handle_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        """Handle text completion request (non-streaming)."""
        self._init_engine()
        
        # Handle prompt
        if isinstance(request.prompt, list):
            prompt = request.prompt[0] if request.prompt else ""
        else:
            prompt = request.prompt
        
        # Create sampling parameters
        sampling_params = self._create_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        
        # Generate
        outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
        output = outputs[0]
        
        # Handle echo
        text = output["text"]
        if request.echo:
            text = prompt + text
        
        # Calculate tokens
        prompt_tokens = len(self.engine.tokenizer.encode(prompt))
        completion_tokens = len(output["token_ids"])
        
        # Create response
        choice = CompletionResponseChoice(
            index=0,
            text=text,
            finish_reason="stop",
        )
        
        response = CompletionResponse(
            model=request.model,
            choices=[choice],
            usage=self._calculate_usage(prompt_tokens, completion_tokens),
        )
        
        return response
    
    async def handle_completion_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[str]:
        """Handle streaming completion request."""
        self._init_engine()
        
        # Handle prompt
        if isinstance(request.prompt, list):
            prompt = request.prompt[0] if request.prompt else ""
        else:
            prompt = request.prompt
        
        # Create sampling parameters
        sampling_params = self._create_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        
        # Generate
        prompt_tokens, completion_tokens = 0, 0 
        self.engine.add_request(prompt, sampling_params)
        while not self.engine.is_finished():
            token_ids, num_token = self.engine.stream_step()
            if num_token > 0:
                prompt_tokens += num_token
            else:
                completion_tokens += abs(num_token)
            # Process finished sequences
            text = self.engine.tokenizer.decode(token_ids)
            chunk = CompletionStreamResponse(
                model=request.model,
                choices=[
                    CompletionStreamResponseChoice(
                        index=0,
                        text=text,
                        finish_reason=None,
                        usage=self._calculate_usage(prompt_tokens, completion_tokens),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk
        final_chunk = CompletionStreamResponse(
            model=request.model,
            choices=[
                CompletionStreamResponseChoice(
                    index=0,
                    text="",
                    finish_reason="stop",
                    usage=self._calculate_usage(prompt_tokens, completion_tokens),
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


def create_app(server: OpenAIServer) -> FastAPI:
    """Create FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        yield
        # Shutdown
        if server.engine is not None:
            server.engine.exit()
    
    app = FastAPI(
        title="NanoVLLM OpenAI API",
        description="OpenAI-compatible API server for nanovllm",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        model_card = ModelCard(id=server.model_name)
        return ModelList(data=[model_card])
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completion endpoint."""
        try:
            if request.stream:
                return StreamingResponse(
                    server.handle_chat_completion_stream(request),
                    media_type="text/event-stream",
                )
            else:
                response = await server.handle_chat_completion(request)
                return JSONResponse(content=response.model_dump())
        except Exception as e:
            error = ErrorResponse(
                message=str(e),
                type="internal_error",
                code=500,
            )
            return JSONResponse(content=error.model_dump(), status_code=500)
    
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """Text completion endpoint."""
        try:
            if request.stream:
                return StreamingResponse(
                    server.handle_completion_stream(request),
                    media_type="text/event-stream",
                )
            else:
                response = await server.handle_completion(request)
                return JSONResponse(content=response.model_dump())
        except Exception as e:
            error = ErrorResponse(
                message=str(e),
                type="internal_error",
                code=500,
            )
            return JSONResponse(content=error.model_dump(), status_code=500)
    
    return app


def run_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    **engine_kwargs,
):
    """
    Run the OpenAI-compatible API server.
    
    Args:
        model: Path to the model
        host: Host to bind to
        port: Port to bind to
        **engine_kwargs: Additional arguments for LLM engine
    """
    server = OpenAIServer(model, **engine_kwargs)
    app = create_app(server)
    
    print(f"Starting nanovllm OpenAI API server on {host}:{port}")
    print(f"Model: {model}")
    print(f"API docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NanoVLLM OpenAI API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Maximum number of sequences")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    engine_kwargs = {
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    
    run_server(
        model=args.model,
        host=args.host,
        port=args.port,
        **engine_kwargs,
    )
