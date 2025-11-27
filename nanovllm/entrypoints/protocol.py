"""
OpenAI API protocol definitions for nanovllm.
Simplified version compatible with OpenAI API format.
"""
import time
from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field


def generate_id(prefix: str) -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}-{int(time.time() * 1000)}"


# ============== Common Models ==============

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorResponse(BaseModel):
    """Error response format."""
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[int] = None


# ============== Model List ==============

class ModelCard(BaseModel):
    """Model card information."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nanovllm"


class ModelList(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# ============== Chat Completion ==============

class ChatMessage(BaseModel):
    """Chat message format."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 64
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    
    # Extra parameters
    ignore_eos: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    """Single choice in chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(default_factory=lambda: generate_id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Delta message for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """Single choice in streaming response."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response."""
    id: str = Field(default_factory=lambda: generate_id("chatcmpl"))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]


# ============== Text Completion ==============

class CompletionRequest(BaseModel):
    """Text completion request."""
    model: str = ""
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 64
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    
    # Extra parameters
    ignore_eos: Optional[bool] = False
    echo: Optional[bool] = False


class CompletionResponseChoice(BaseModel):
    """Single choice in completion response."""
    index: int
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    """Text completion response."""
    id: str = Field(default_factory=lambda: generate_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(BaseModel):
    """Single choice in streaming completion response."""
    index: int
    text: str
    usage: UsageInfo
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    """Streaming completion response."""
    id: str = Field(default_factory=lambda: generate_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionStreamResponseChoice]
