"""
LightRAG LLM Provider Selector
Hỗ trợ các provider: OpenAI, Azure OpenAI, Ollama, Google Gemini, Hugging Face
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from functools import partial


class LightRAGLLMSelector:
    """
    Class để chọn và cấu hình LLM provider cho LightRAG
    
    Attributes:
        provider (str): Tên provider (openai, azure, ollama, gemini, huggingface)
        api_key (str): API key cho provider
        llm_model_name (str): Tên model LLM
        embedding_model_name (str): Tên model embedding
        additional_config (dict): Cấu hình bổ sung cho provider
    """
    
    SUPPORTED_PROVIDERS = {
        'openai': {
            'default_llm_model': 'gpt-4o-mini',
            'default_embed_model': 'text-embedding-3-large',
            'default_embed_dim': 3072,
            'default_max_tokens': 8192
        },
        'azure': {
            'default_llm_model': 'gpt-4o-mini',
            'default_embed_model': 'text-embedding-3-large',
            'default_embed_dim': 1536,
            'default_max_tokens': 8192
        },
        'ollama': {
            'default_llm_model': 'llama3.2',
            'default_embed_model': 'nomic-embed-text',
            'default_embed_dim': 768,
            'default_max_tokens': 8192
        },
        'gemini': {
            'default_llm_model': 'gemini-2.0-flash',
            'default_embed_model': 'models/text-embedding-004',
            'default_embed_dim': 768,
            'default_max_tokens': 2048
        },
        'huggingface': {
            'default_llm_model': 'meta-llama/Llama-3.1-8B-Instruct',
            'default_embed_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'default_embed_dim': 384,
            'default_max_tokens': 2048
        }
    }
    
    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Khởi tạo LLM selector
        
        Args:
            provider: Tên provider (openai, azure, ollama, gemini, huggingface)
            api_key: API key (không bắt buộc với ollama và huggingface)
            llm_model_name: Tên model LLM (sử dụng default nếu không cung cấp)
            embedding_model_name: Tên model embedding (sử dụng default nếu không cung cấp)
            **kwargs: Các tham số bổ sung như base_url, azure_endpoint, etc.
        """
        self.provider = provider.lower()
        
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' không được hỗ trợ. "
                f"Các provider hỗ trợ: {list(self.SUPPORTED_PROVIDERS.keys())}"
            )
        
        provider_config = self.SUPPORTED_PROVIDERS[self.provider]
        
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        self.llm_model_name = llm_model_name or provider_config['default_llm_model']
        self.embedding_model_name = embedding_model_name or provider_config['default_embed_model']
        self.embedding_dim = kwargs.get('embedding_dim', provider_config['default_embed_dim'])
        self.max_token_size = kwargs.get('max_token_size', provider_config['default_max_tokens'])
        self.additional_config = kwargs
        
        # Validate API key cho các provider cần API key
        if self.provider in ['openai', 'azure', 'gemini'] and not self.api_key:
            raise ValueError(f"API key là bắt buộc cho provider '{self.provider}'")
    
    def get_llm_model_func(self):
        """
        Trả về LLM model function dựa trên provider
        
        Returns:
            callable: LLM model function
        """
        if self.provider == 'openai':
            return self._get_openai_llm_func()
        elif self.provider == 'azure':
            return self._get_azure_llm_func()
        elif self.provider == 'ollama':
            return self._get_ollama_llm_func()
        elif self.provider == 'gemini':
            return self._get_gemini_llm_func()
        elif self.provider == 'huggingface':
            return self._get_huggingface_llm_func()
        else:
            raise NotImplementedError(f"Provider {self.provider} chưa được implement")
    
    def get_embedding_func(self):
        """
        Trả về embedding function dựa trên provider
        
        Returns:
            callable: Embedding function được wrap với attributes
        """
        if self.provider == 'openai':
            return self._get_openai_embedding_func()
        elif self.provider == 'azure':
            return self._get_azure_embedding_func()
        elif self.provider == 'ollama':
            return self._get_ollama_embedding_func()
        elif self.provider == 'gemini':
            return self._get_gemini_embedding_func()
        elif self.provider == 'huggingface':
            return self._get_huggingface_embedding_func()
        else:
            raise NotImplementedError(f"Provider {self.provider} chưa được implement")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Trả về dict config đầy đủ cho LightRAG
        
        Returns:
            dict: Config cho LightRAG instance
        """
        return {
            'llm_model_func': self.get_llm_model_func(),
            'embedding_func': self.get_embedding_func(),
            'llm_model_name': self.llm_model_name,
        }
    
    # OpenAI Implementation
    def _get_openai_llm_func(self):
        """OpenAI LLM function"""
        from lightrag.llm.openai import openai_complete_if_cache
        
        base_url = self.additional_config.get('base_url')
        
        async def llm_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            **kwargs
        ):
            return await openai_complete_if_cache(
                self.llm_model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                base_url=base_url,
                **kwargs
            )
        
        return llm_func
    
    def _get_openai_embedding_func(self):
        """OpenAI embedding function"""
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        
        base_url = self.additional_config.get('base_url')
        
        @wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.max_token_size,
            model_name=self.embedding_model_name
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await openai_embed.func(
                texts,
                model=self.embedding_model_name,
                api_key=self.api_key,
                base_url=base_url
            )
        
        return embedding_func
    
    # Azure OpenAI Implementation
    def _get_azure_llm_func(self):
        """Azure OpenAI LLM function"""
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache
        
        azure_endpoint = self.additional_config.get('azure_endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = self.additional_config.get('api_version') or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        deployment_name = self.additional_config.get('deployment_name') or os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        async def llm_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            **kwargs
        ):
            return await azure_openai_complete_if_cache(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment_name=deployment_name,
                **kwargs
            )
        
        return llm_func
    
    def _get_azure_embedding_func(self):
        """Azure OpenAI embedding function"""
        from lightrag.llm.azure_openai import azure_openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        
        azure_endpoint = self.additional_config.get('azure_endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = self.additional_config.get('api_version') or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        embedding_deployment = self.additional_config.get('embedding_deployment') or os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')
        
        @wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.max_token_size,
            model_name=self.embedding_model_name
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await azure_openai_embed.func(
                texts,
                api_key=self.api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment_name=embedding_deployment
            )
        
        return embedding_func
    
    # Ollama Implementation
    def _get_ollama_llm_func(self):
        """Ollama LLM function"""
        from lightrag.llm.ollama import ollama_model_complete
        
        ollama_host = self.additional_config.get('ollama_host', 'http://localhost:11434')
        
        async def llm_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            **kwargs
        ):
            return await ollama_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                model=self.llm_model_name,
                host=ollama_host,
                options={"num_ctx": 32768},  # Tăng context size
                **kwargs
            )
        
        return llm_func
    
    def _get_ollama_embedding_func(self):
        """Ollama embedding function"""
        from lightrag.llm.ollama import ollama_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        
        ollama_host = self.additional_config.get('ollama_host', 'http://localhost:11434')
        
        @wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.max_token_size,
            model_name=self.embedding_model_name
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await ollama_embed.func(
                texts,
                embed_model=self.embedding_model_name,
                host=ollama_host
            )
        
        return embedding_func
    
    # Google Gemini Implementation
    def _get_gemini_llm_func(self):
        """Google Gemini LLM function"""
        from lightrag.llm.gemini import gemini_model_complete
        
        async def llm_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            **kwargs
        ):
            return await gemini_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                model_name=self.llm_model_name,
                **kwargs
            )
        
        return llm_func
    
    def _get_gemini_embedding_func(self):
        """Google Gemini embedding function"""
        from lightrag.llm.gemini import gemini_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        
        @wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.max_token_size,
            model_name=self.embedding_model_name
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await gemini_embed.func(
                texts,
                api_key=self.api_key,
                model=self.embedding_model_name
            )
        
        return embedding_func
    
    # Hugging Face Implementation
    def _get_huggingface_llm_func(self):
        """Hugging Face LLM function"""
        from lightrag.llm.hf import hf_model_complete
        
        async def llm_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            **kwargs
        ):
            return await hf_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                model_name=self.llm_model_name,
                **kwargs
            )
        
        return llm_func
    
    def _get_huggingface_embedding_func(self):
        """Hugging Face embedding function"""
        from transformers import AutoTokenizer, AutoModel
        from lightrag.llm.hf import hf_embed
        from lightrag.utils import EmbeddingFunc
        
        # Pre-load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        embed_model = AutoModel.from_pretrained(self.embedding_model_name)
        
        embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=self.max_token_size,
            model_name=self.embedding_model_name,
            func=partial(
                hf_embed.func,
                tokenizer=tokenizer,
                embed_model=embed_model
            )
        )
        
        return embedding_func


# Hàm tiện ích để tạo LightRAG instance
def create_lightrag_with_provider(
    provider: str,
    api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    working_dir: str = "./lightrag_cache",
    **kwargs
):
    """
    Hàm tiện ích để tạo LightRAG instance với provider cụ thể
    
    Args:
        provider: Tên provider
        api_key: API key
        llm_model_name: Tên LLM model
        embedding_model_name: Tên embedding model
        working_dir: Thư mục lưu cache
        **kwargs: Các config bổ sung
        
    Returns:
        LightRAG instance
    """
    from lightrag import LightRAG
    
    selector = LightRAGLLMSelector(
        provider=provider,
        api_key=api_key,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        **kwargs
    )
    
    config = selector.get_config()
    
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=config['llm_model_func'],
        embedding_func=config['embedding_func'],
        llm_model_name=config['llm_model_name']
    )
    
    return rag


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example 1: OpenAI
        print("=== OpenAI Example ===")
        selector_openai = LightRAGLLMSelector(
            provider="openai",
            api_key="your-openai-api-key",
            llm_model_name="gpt-4o-mini"
        )
        config = selector_openai.get_config()
        print(f"LLM Model: {selector_openai.llm_model_name}")
        print(f"Embedding Model: {selector_openai.embedding_model_name}")
        
        # Example 2: Ollama (local, không cần API key)
        print("\n=== Ollama Example ===")
        selector_ollama = LightRAGLLMSelector(
            provider="ollama",
            llm_model_name="llama3.2",
            embedding_model_name="nomic-embed-text"
        )
        config = selector_ollama.get_config()
        print(f"LLM Model: {selector_ollama.llm_model_name}")
        print(f"Embedding Model: {selector_ollama.embedding_model_name}")
        
        # Example 3: Google Gemini
        print("\n=== Gemini Example ===")
        selector_gemini = LightRAGLLMSelector(
            provider="gemini",
            api_key="your-gemini-api-key",
            llm_model_name="gemini-2.0-flash"
        )
        config = selector_gemini.get_config()
        print(f"LLM Model: {selector_gemini.llm_model_name}")
        print(f"Embedding Model: {selector_gemini.embedding_model_name}")
        
        # Example 4: Sử dụng hàm helper để tạo LightRAG instance
        print("\n=== Create LightRAG Instance ===")
        # rag = create_lightrag_with_provider(
        #     provider="openai",
        #     api_key="your-api-key",
        #     working_dir="./rag_storage"
        # )
        # await rag.initialize_storages()
        # print("LightRAG instance created successfully!")
    
    asyncio.run(main())
