import httpx
import json
import logging
import re
import asyncio
from typing import List, Dict, Optional
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
    
    async def _get_price_boundaries(self, tenant_id: str) -> Optional[Dict]:
        """Получение динамических ценовых границ для тенанта"""
        try:
            price_query = """
                SELECT 
                    MIN(price_int) as min_price,
                    MAX(price_int) as max_price
                FROM flbot.products 
                WHERE tenant_id = $1 AND stock = TRUE
            """
            
            if not db_service.pool:
                logger.warning("Database pool not initialized")
                return None
                
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
            if not db_service.pool:
                logger.warning("Database pool not initialized")
                return {}
                
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
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Извлечение JSON из ответа AI модели"""
        try:
            cleaned_text = response_text.strip()
            
            # Remove markdown code block markers
            if "```json" in cleaned_text:
                cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
                cleaned_text = re.sub(r'\s*```', '', cleaned_text)
            elif "```" in cleaned_text:
                cleaned_text = re.sub(r'```', '', cleaned_text)
            
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
    
    def _normalize_azerbaijani(self, text: str) -> str:
        """Нормализация азербайджанского текста для поддержки транслитерации"""
        replacements = {
            'ə': 'e', 'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ə': 'E', 'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
        }
        normalized = text
        for original, replacement in replacements.items():
            normalized = normalized.replace(original, replacement)
        return normalized.lower()
    
    async def _create_fallback_classification(self, query: str, tenant_id: str, language: str = "ru") -> Dict:
        """Улучшенный языково-независимый fallback с поддержкой мультиязычности и транслитерации"""
        query_lower = query.lower()
        if language == "az":
            query_lower = self._normalize_azerbaijani(query)
        
        # Получаем ценовые границы
        price_boundaries = await self._get_price_boundaries(tenant_id)
        
        # Логирование входных данных
        logger.debug(f"Fallback classification for query: {query}, language: {language}, tenant: {tenant_id}")
        
        # Извлекаем числа из запроса
        numbers = re.findall(r'\d+', query)
        logger.debug(f"Extracted numbers: {numbers}")
        
        # Языково-независимая числовая логика
        if len(numbers) >= 2:
            num1, num2 = int(numbers[0]), int(numbers[1])
            
            # Паттерн: маленькое количество + большой бюджет
            if (2 <= num1 <= 10 and num2 > 50 and num2 > num1 * 15):
                logger.info(f"Fallback detected BUDGET_SPLIT: {num1} items for {num2} budget")
                return {
                    "type": "BUDGET_SPLIT",
                    "confidence": 0.6,
                    "parameters": {
                        "quantity": num1,
                        "max_budget": num2,
                        "sort_order": None,
                        "min_price": None,
                        "max_price": None
                    }
                }
            
            # Паттерн: диапазон цен (два близких числа)
            elif (abs(num1 - num2) < max(num1, num2) and min(num1, num2) > 5):
                logger.info(f"Fallback detected price range: {min(num1, num2)}-{max(num1, num2)}")
                return {
                    "type": "PRODUCT_SEARCH",
                    "confidence": 0.7,
                    "parameters": {
                        "quantity": None,
                        "max_budget": None,
                        "sort_order": None,
                        "min_price": min(num1, num2),
                        "max_price": max(num1, num2)
                    }
                }
        
        # Мультиязычные паттерны
        currency_patterns = {
            "ru": ['манат', 'azn', 'рубль', 'руб', 'доллар', 'usd', 'евро', 'eur'],
            "en": ['manat', 'azn', 'ruble', 'dollar', 'usd', 'euro', 'eur'],
            "az": ['manat', 'azn', 'rubl', 'dollar', 'avro']
        }
        
        exact_price_patterns = {
            "ru": [
                r'за\s*(\d+)\s*({})',
                r'по\s*цене\s*(\d+)\s*({})',
                r'стоимость\s*(\d+)\s*({})',
                r'цена\s*(\d+)\s*({})'
            ],
            "en": [
                r'for\s*(\d+)\s*({})',
                r'at\s*price\s*(\d+)\s*({})',
                r'cost\s*(\d+)\s*({})',
                r'price\s*(\d+)\s*({})'
            ],
            "az": [
                r'ucun\s*(\d+)\s*({})',
                r'qiymete\s*(\d+)\s*({})',
                r'xerc\s*(\d+)\s*({})',
                r'qiymet\s*(\d+)\s*({})',
                # Транслитерация
                r'uchun\s*(\d+)\s*({})',
                r'qiymata\s*(\d+)\s*({})',
                r'xarj\s*(\d+)\s*({})'
            ]
        }
        
        max_price_patterns = {
            "ru": [
                r'до\s*(\d+)\s*({})',
                r'максимум\s*(\d+)\s*({})',
                r'не\s*дороже\s*(\d+)\s*({})'
            ],
            "en": [
                r'up\s*to\s*(\d+)\s*({})',
                r'maximum\s*(\d+)\s*({})',
                r'not\s*more\s*than\s*(\d+)\s*({})'
            ],
            "az": [
                r'qeder\s*(\d+)\s*({})',
                r'maksimum\s*(\d+)\s*({})',
                r'daha\s*bahali\s*deyil\s*(\d+)\s*({})',
                # Транслитерация
                r'qadar\s*(\d+)\s*({})',
                r'maximum\s*(\d+)\s*({})'
            ]
        }
        
        price_range_pattern = {
            "ru": r'от\s*(\d+)\s*до\s*(\d+)\s*({})',
            "en": r'from\s*(\d+)\s*to\s*(\d+)\s*({})',
            "az": [
                r'dan\s*(\d+)\s*dek\s*(\d+)\s*({})',
                r'dan\s*(\d+)\s*dak\s*(\d+)\s*({})'  # Транслитерация
            ]
        }
        
        extreme_keywords = {
            "ru": ['самый дорогой', 'самый дешевый', 'дороже всего', 'дешевле всего', 'максимальн', 'минимальн'],
            "en": ['most expensive', 'cheapest', 'highest price', 'lowest price', 'maximum', 'minimum'],
            "az": ['en bahali', 'en ucuz', 'en yuksek qiymet', 'en asagi qiymet', 'maksimum', 'minimum',
                   'en bahaly', 'en ucuz', 'en yuksek qiymat', 'en asagy qiymat']  # Транслитерация
        }
        
        cheap_keywords = {
            "ru": ['недорог', 'дешев'],
            "en": ['cheap', 'inexpensive', 'affordable'],
            "az": ['ucuz', 'qenaetcil', 'ucuz']  # Транслитерация
        }
        
        expensive_keywords = {
            "ru": ['дорог', 'премиум'],
            "en": ['expensive', 'premium', 'luxury'],
            "az": ['bahali', 'premium', 'bahaly']  # Транслитерация
        }
        
        # Обработка точных цен
        if language in exact_price_patterns:
            for pattern in (price_range_pattern[language] if isinstance(price_range_pattern[language], list) else [price_range_pattern[language]]):
                pattern = pattern.format('|'.join(currency_patterns[language]))
                match = re.search(pattern, query_lower)
                if match:
                    logger.info(f"Fallback detected price range: {match.group(1)}-{match.group(2)}")
                    return {
                        "type": "PRODUCT_SEARCH",
                        "confidence": 0.8,
                        "parameters": {
                            "quantity": None,
                            "max_budget": None,
                            "sort_order": None,
                            "min_price": int(match.group(1)),
                            "max_price": int(match.group(2))
                        }
                    }
        
        # Максимальные цены
        if language in max_price_patterns:
            for pattern in max_price_patterns[language]:
                pattern = pattern.format('|'.join(currency_patterns[language]))
                match = re.search(pattern, query_lower)
                if match:
                    price = int(match.group(1))
                    logger.info(f"Fallback detected max price: {price}")
                    return {
                        "type": "PRODUCT_SEARCH",
                        "confidence": 0.8,
                        "parameters": {
                            "quantity": None,
                            "max_budget": None,
                            "sort_order": None,
                            "min_price": None,
                            "max_price": price
                        }
                    }
        
        # Динамические границы для "недорогие"/"дорогие"
        if price_boundaries:
            min_boundary = price_boundaries["min_price"]
            max_boundary = price_boundaries["max_price"]
            
            if language in cheap_keywords and any(keyword in query_lower for keyword in cheap_keywords[language]):
                logger.info(f"Fallback detected cheap products, max price: {min_boundary}")
                return51                    "type": "PRODUCT_SEARCH",
                    "confidence": 0.7,
                    "parameters": {
                        "quantity": None,
                        "max_budget": None,
                        "sort_order": None,
                        "min_price": None,
                        "max_price": min_boundary
                    }
                }
            elif language in expensive_keywords and any(keyword in query_lower for keyword in expensive_keywords[language]):
                logger.info(f"Fallback detected expensive products, min price: {max_boundary}")
                return {
                    "type": "PRODUCT_SEARCH",
                    "confidence": 0.7,
                    "parameters": {
                        "quantity": None,
                        "max_budget": None,
                        "sort_order": None,
                        "min_price": max_boundary,
                        "max_price": None
                    }
                }
        
        # Экстремальные цены
        if language in extreme_keywords and any(keyword in query_lower for keyword in extreme_keywords[language]):
            sort_order = "DESC" if any(word in query_lower for word in expensive_keywords.get(language, [])) else "ASC"
            logger.info(f"Fallback detected extreme price, sort order: {sort_order}")
            return {
                "type": "FIND_EXTREME_PRICE",
                "confidence": 0.7,
                "parameters": {
                    "quantity": None,
                    "max_budget": None,
                    "sort_order": sort_order,
                    "min_price": None,
                    "max_price": None
                }
            }
        
        # Default fallback
        logger.info(f"Using default fallback for language: {language}")
        return {
            "type": "PRODUCT_SEARCH",
            "confidence": 0.3,
            "parameters": {
                "quantity": None,
                "max_budget": None,
                "sort_order": None,
                "min_price": None,
                "max_price": None
            }
        }
    
    async def classify_query(self, query: str, tenant_id: str, language: str = "ru") -> Dict:
        """AI-first классификация с мультиязычной поддержкой"""
        try:
            # Получаем конфигурацию тенанта
            config = None
            if db_service.pool:
                try:
                    config = await db_service.get_tenant_config(tenant_id)
                except Exception as e:
                    logger.warning(f"Error getting tenant config: {e}")
            
            # Если нет промпта - используем fallback
            if not config or not config.get("classification_prompt"):
                logger.warning(f"No classification prompt for tenant {tenant_id}, using fallback")
                return await self._create_fallback_classification(query, tenant_id, language)
            
            logger.debug(f"Classifying query for tenant {tenant_id}: {query[:100]}... (language: {language})")
            
            # Получаем дополнительные данные
            tenant_context = await self._get_tenant_context(tenant_id)
            price_boundaries = await self._get_price_boundaries(tenant_id)
            
            # Формируем промпт
            base_prompt = config['classification_prompt']
            
            # Добавляем контекст если есть данные
            context_addition = ""
            currency = tenant_context.get('currency', 'USD') if tenant_context else 'USD'
            if tenant_context:
                context_addition = (
                    f"\n\nTENANT CONTEXT:\n"
                    f"- Business: {tenant_context.get('business_type', 'retail')}\n"
                    f"- Location: {tenant_context.get('location', 'N/A')}\n"
                    f"- Currency: {currency}"
                )
            
            if price_boundaries:
                context_addition += (
                    f"\n\nPRICE RANGE: {price_boundaries['min_price']}-{price_boundaries['max_price']} {currency}"
                )
            else:
                context_addition += (
                    f"\n\nPRICE RANGE: Use default values (min: 25, max: 100) {currency}"
                )
            
            # Формируем финальный промпт с безопасным форматированием
            enhanced_prompt = (
                f"{base_prompt}\n\n"
                f"{context_addition}\n\n"
                f"USER QUERY: {query!r}\n"
                f"USER LANGUAGE: {language}\n\n"
                f"Analyze semantic meaning regardless of specific words. "
                f"Consider the tenant's currency ({currency}) when interpreting prices. "
                f"Handle transliterated text for Azerbaijani (e.g., 'cicek' instead of 'çicək'). "
                f"Return JSON classification with type (PRODUCT_SEARCH|BUDGET_SPLIT|FIND_EXTREME_PRICE), "
                f"confidence (0.0-1.0), and parameters (quantity, max_budget, sort_order, min_price, max_price)."
            )
            
            # Вызываем AI
            response = await asyncio.wait_for(
                self.gemini_client.generate_content_async(enhanced_prompt),
                timeout=30
            )
            
            if response and response.text:
                result = self._extract_json_from_response(response.text)
                
                if result and isinstance(result, dict):
                    # Валидация
                    if "type" not in result or result["type"] not in ["PRODUCT_SEARCH", "BUDGET_SPLIT", "FIND_EXTREME_PRICE"]:
                        logger.warning(f"Invalid classification type: {result.get('type')}")
                        return await self._create_fallback_classification(query, tenant_id, language)
                    
                    # Нормализация параметров
                    if "parameters" not in result:
                        result["parameters"] = {}
                    
                    result["parameters"] = {
                        "quantity": result["parameters"].get("quantity"),
                        "max_budget": result["parameters"].get("max_budget"),
                        "sort_order": result["parameters"].get("sort_order"),
                        "min_price": result["parameters"].get("min_price"),
                        "max_price": result["parameters"].get("max_price")
                    }
                    
                    if "confidence" not in result:
                        result["confidence"] = 0.8
                    
                    logger.info(f"AI classified as: {result['type']} (confidence: {result.get('confidence', 0)})")
                    return result
                else:
                    logger.warning("Failed to extract valid JSON from AI response")
                    return await self._create_fallback_classification(query, tenant_id, language)
            else:
                logger.warning("Empty response from AI")
                return await self._create_fallback_classification(query, tenant_id, language)
                
        except asyncio.TimeoutError:
            logger.error("AI classification timed out")
            return await self._create_fallback_classification(query, tenant_id, language)
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
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
                
                embedding = data["embeddings"][0]
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