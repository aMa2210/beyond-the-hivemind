import os
import openai
import tiktoken


class GPTModel:
    def __init__(self, config):
        self.model_name = config["model_name"]
        self.num_tokens = config.get("num_tokens", 1024)
        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 0.9)
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def batch_generate(self, prompts):
        results = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.num_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            results.append(response.choices[0].message.content)
        return results


_EMBED_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_EMBED_TOKENS = 8000  # leave buffer below OpenAI's 8192-token hard limit


def _truncate_to_token_limit(text: str) -> str:
    tokens = _EMBED_ENCODING.encode(text)
    if len(tokens) > _MAX_EMBED_TOKENS:
        return _EMBED_ENCODING.decode(tokens[:_MAX_EMBED_TOKENS])
    return text


def embed_texts(texts, model="text-embedding-3-small", batch_size=512):
    """Embed a list of texts using OpenAI embedding API. Returns list of vectors."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    texts = [_truncate_to_token_limit(t) for t in texts]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def embed_texts_gemini(texts, model="gemini-embedding-2", max_concurrent=10):
    """Embed texts using Gemini API with async concurrency. Returns list of vectors."""
    import asyncio
    import random
    from google import genai
    from google.genai.errors import ClientError, ServerError

    async def _run():
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(text):
            async with sem:
                for attempt in range(6):
                    try:
                        r = await client.aio.models.embed_content(model=model, contents=text)
                        return list(r.embeddings[0].values)
                    except (ClientError, ServerError) as e:
                        retryable = isinstance(e, ServerError) or "429" in str(e)
                        if retryable and attempt < 5:
                            wait = (2 ** attempt) + random.random()
                            print(f"[{str(e)[:20]}] Retrying in {wait:.0f}s (attempt {attempt+1}/6)")
                            await asyncio.sleep(wait)
                        else:
                            raise

        return await asyncio.gather(*[_one(t) for t in texts])

    return list(asyncio.run(_run()))
