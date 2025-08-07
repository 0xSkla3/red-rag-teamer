#app/factories.py
from .chunk_strategies import (
    SemanticChunkStrategy, TopicChunkStrategy, AstChunkStrategy,
    JsonChunkStrategy, LogChunkStrategy, SlidingWindowStrategy,
    CompositeChunkStrategy, HierarchicalChunkStrategy, LateChunkingDecorator
)
from .chunk_handlers import (
    HierarchicalChunkHandler, CodeChunkHandler, JsonChunkHandler, 
    LogChunkHandler, TextChunkHandler, ExploitPayloadHandler, AssemblyHandler
)
from .pipeline import BaseChunkPipeline

class ChunkStrategyFactory:
    @staticmethod
    def get_strategy(name: str, **kwargs) -> IChunkStrategy:
        strategies = {
            'semantic': lambda: SemanticChunkStrategy(
                kwargs.get('model_name', 'all-MiniLM-L6-v2'),
                kwargs.get('device', 'cpu'),
                kwargs.get('similarity_threshold', 0.75)
            ),
            'ast': lambda: AstChunkStrategy(kwargs.get('language', 'python')),
            'json': JsonChunkStrategy,
            'log': lambda: LogChunkStrategy(kwargs.get('timestamp_pattern')),
            'hierarchical': lambda: HierarchicalChunkStrategy(
                kwargs.get('header_patterns', ['^# ', '^## ', '^### '])
            ),
            'sliding': lambda: SlidingWindowStrategy(
                kwargs.get('window_size', 512),
                kwargs.get('overlap', 0.1)
            ),
            'composite': lambda: CompositeChunkStrategy(
                [ChunkStrategyFactory.get_strategy(s) for s in kwargs.get('strategies', [])]
            ),
            'late': lambda: LateChunkingDecorator(
                kwargs['wrapped'],
                kwargs.get('context_model_name', 'all-mpnet-base-v2'),
                kwargs.get('device', 'cpu')
            )
        }
        return strategies[name]()

class ChunkPipelineBuilder:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler: ChunkHandler) -> 'ChunkPipelineBuilder':
        self.handlers.append(handler)
        return self

    def build(self) -> BaseChunkPipeline:
        chain = None
        for handler in reversed(self.handlers):
            if chain:
                handler.next = chain
            chain = handler
        return BaseChunkPipeline(chain)