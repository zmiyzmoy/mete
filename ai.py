import httpx
import json
import logging
import re
import asyncio
from typing import List, Dict
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from services.redis_client import redis_service
from services.database import db_service

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.gemini_client = None
    
    async def initialize(self):
        try:
            genai.configure(api_key=settings.google_api_key)
            self.gemini_client = genai.GenerativeModel(
                model_name=settings.gemini_model,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.gemini_temperature
                )
            )
            logger.info(f"Gemini client initialized successfully: {settings.gemini_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    async def _get_price_boundaries(self, tenant_id: str) -> Dict:
        """Получение динамических ценовых границ для тенанта"""
        try:
            price_query = """
                SELECT 
                    MIN(price_int) as min_price,
                    MAX(price_int) as max_price
                FROM flbot.products 
                WHERE tenant_id = $1 AND stock = TRUE
            """
            
            async with db_service.pool.acquire() as conn:
                await conn.execute(f"SET app.tenant_id = '{tenant_id}';")
                result = await conn.fetchrow(price_query, tenant_id)
                
                if result and result['min_price'] and result['max_price']:
                    return {
                        "min_price": int(result['min_price'] / 100),
                        "max_price": int(result['max_price'] / 100)
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting price boundaries for tenant {tenant_id}: {e}")
            return None
    
    async def _get_tenant_context(self, tenant_id: str) -> Dict:
        """Получение контекста тенанта из базы данных"""
        try:
            query = """
                SELECT business_type, location, currency 
                FROM flbot.tenants 
                WHERE tenant_id = $1
            """
            
            async with db_service.pool.acquire() as conn:
                result = await conn.fetchrow(query, tenant_id)
                return dict(result) if result else {}
                
        except Exception as e:
            logger.error(f"Error getting tenant context for {tenant_id}: {e}")
            return {}
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        try:
            cleaned_text = response_text.strip()
            
            # Remove markdown code block markers
            if "```
                cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
                cleaned_text = re.sub(r'\s*```
            elif "```" in cleaned_text:
                cleaned_text = re.sub(r'```
            
            # Extract JSON content
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                if isinstance(result, dict) and "type" in result:
                    return result
            
            # Try parsing the entire cleaned text as JSON
            result = json.loads(cleaned_text)
            return result
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, text: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}, text: {response_text}")
            return None
    
    async def _create_fallback_classification(self, query: str, tenant_id: str, language: str = "ru") -> Dict:
        """Language-agnostic numeric fallback"""
        
        # Извлекаем только числа - работает универсально для любого языка
        numbers = re.findall(r'\d+', query)
        
        if len(numbers) == 2:
            num1, num2 = int(numbers), int(numbers)
            
            # BUDGET_SPLIT: строгие условия для высокой точности
            if (2 <= num1 <= 10 and      # разумное количество
                num2 > 50 and            # значительный бюджет
                num2 > num1 * 15):       # бюджет много больше количества
                
                logger.info(f"Fallback detected BUDGET_SPLIT: {num1} items for {num2} budget")
                return {
                    "type": "BUDGET_SPLIT",
                    "confidence": 0.4,
                    "parameters": {
                        "quantity": num1,
                        "max_budget": num2,
                        "sort_order": None,
                        "min_price": None,
                        "max_price": None
                    }
                }
            
            # PRICE_RANGE: два числа близкого порядка  
            elif (abs(num1 - num2) < max(num1, num2) * 2 and
                  min(num1, num2) > 10):
                
                logger.info(f"Fallback detected price range: {min(num1, num2)}-{max(num1, num2)}")
                return {
                    "type": "PRODUCT_SEARCH", 
                    "confidence": 0.5,
                    "parameters": {
                        "quantity": None,
                        "max_budget": None,
                        "sort_order": None,
                        "min_price": min(num1, num2),
                        "max_price": max(num1, num2)
                    }
                }
        
        # Default fallback - никаких языковых предположений
        logger.info("Using default fallback classification")
        return {
            "type": "PRODUCT_SEARCH",
            "confidence": 0.2,
            "parameters": {
                "quantity": None,
                "max_budget": None,
                "sort_order": None,
                "min_price": None,
                "max_price": None
            }
        }
    
    async def classify_query(self, query: str, tenant_id: str, language: str = "ru") -> Dict:
        """AI-first classification with language support and safe fallback"""
        try:
            config = await db_service.get_tenant_config(tenant_id)
            base_prompt = config.get('classification_prompt') if config else None
            
            if not base_prompt:
                logger.warning(f"No classification prompt for tenant {tenant_id}")
                return await self._create_fallback_classification(query, tenant_id, language)
            
            tenant_context = await self._get_tenant_context(tenant_id)
            price_boundaries = await self._get_price_boundaries(tenant_id)
            
            # Собираем ВСЕ необходимые данные для промпта
            format_data = {
                'query': query,
                'language': language,
                'business_type': tenant_context.get('business_type') if tenant_context else None,
                'location': tenant_context.get('location') if tenant_context else None,
                'currency': tenant_context.get('currency') if tenant_context else None,
                'min_price': price_boundaries.get('min_price') if price_boundaries else None,
                'max_price': price_boundaries.get('max_price') if price_boundaries else None
            }
            
            # Проверяем что все необходимые данные есть
            missing_fields = [key for key, value in format_data.items() 
                             if value is None and key not in ['query', 'language']]
            
            if missing_fields:
                logger.warning(f"Missing data for tenant {tenant_id}: {missing_fields}")
                return await self._create_fallback_classification(query, tenant_id, language)
            
            try:
                # Безопасно подставляем данные в промпт из PostgreSQL
                final_prompt = base_prompt.format(**format_data)
                
                response = await asyncio.wait_for(
                    self.gemini_client.generate_content_async(final_prompt),
                    timeout=30
                )
                
                if response and response.text:
                    result = self._extract_json_from_response(response.text)
                    
                    if result and isinstance(result, dict):
                        if "type" not in result or result["type"] not in ["PRODUCT_SEARCH", "BUDGET_SPLIT", "FIND_EXTREME_PRICE"]:
                            logger.warning(f"Invalid classification type: {result}")
                            return await self._create_fallback_classification(query, tenant_id, language)
                        
                        if "parameters" not in result:
                            result["parameters"] = {}
                        
                        # Нормализуем параметры
                        result["parameters"] = {
                            "quantity": result["parameters"].get("quantity"),
                            "max_budget": result["parameters"].get("max_budget"),
                            "sort_order": result["parameters"].get("sort_order"),
                            "min_price": result["parameters"].get("min_price"),
                            "max_price": result["parameters"].get("max_price")
                        }
                        
                        if "confidence" not in result:
                            result["confidence"] = 0.8
                        
                        logger.info(f"Query classified as: {result['type']} with confidence: {result.get('confidence', 0)}")
                        return result
                    else:
                        logger.warning(f"Failed to extract valid JSON from response: {response.text}")
                        return await self._create_fallback_classification(query, tenant_id, language)
                else:
                    logger.warning("Empty response from Gemini")
                    return await self._create_fallback_classification(query, tenant_id, language)
            
            except KeyError as e:
                logger.error(f"Missing placeholder in prompt template: {e}")
                return await self._create_fallback_classification(query, tenant_id, language)
                
        except asyncio.TimeoutError:
            logger.error("Gemini classification timed out")
            return await self._create_fallback_classification(query, tenant_id, language)
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return await self._create_fallback_classification(query, tenant_id, language)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_cohere_embedding(self, text: str, api_key: str = None) -> List[float]:
        if not api_key:
            api_key = settings.cohere_api_key
        
        logger.debug(f"Getting embedding for text: {text[:100]}...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.cohere.com/v1/embed",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "texts": [text],
                        "model": settings.cohere_model,
                        "input_type": "search_query"
                    },
                    timeout=settings.cohere_timeout
                )
                response.raise_for_status()
                data = response.json()
                
                embedding = data["embeddings"]
                logger.debug(f"Embedding generated successfully, dimensions: {len(embedding)}")
                
                return embedding
                
            except httpx.HTTPError as e:
                logger.error(f"Cohere API HTTP error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in Cohere API: {e}")
                raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def call_gemini_with_memory(self, prompt: str, session_id: str) -> str:
        try:
            logger.debug(f"Calling Gemini with memory for session: {session_id}")
            
            chat_history = await redis_service.get_chat_history(session_id)
            
            context_messages = []
            for msg in chat_history[-10:]:
                context_messages.append(f"{msg['role']}: {msg['content']}")
            
            full_prompt = prompt
            if context_messages:
                full_prompt = f"""
Контекст предыдущих сообщений:
{chr(10).join(context_messages)}

Текущий запрос:
{prompt}
"""
            
            logger.debug(f"Full prompt length: {len(full_prompt)} characters")
            
            response = await asyncio.wait_for(
                self.gemini_client.generate_content_async(full_prompt),
                timeout=45
            )
            
            if response and response.text:
                await redis_service.save_chat_history(session_id, prompt, response.text)
                
                logger.info(f"Gemini response generated successfully for session: {session_id}")
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return "К сожалению, произошла ошибка при генерации ответа."
                
        except asyncio.TimeoutError:
            logger.error("Gemini generation timed out")
            return "К сожалению, генерация ответа заняла слишком много времени. Попробуйте еще раз."
        except Exception as e:
            logger.error(f"Error calling Gemini with memory: {e}")
            return "К сожалению, произошла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."

ai_service = AIService()
