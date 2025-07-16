"""
Unified Product MCP Server - ПОЛНАЯ ВЕРСИЯ
Точная реализация всех workflow без урезаний и заглушек
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import asyncpg
import httpx
import os
import logging
from datetime import datetime
import json
import copy
import asyncio
import google.generativeai as genai
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="Unified Product MCP Server",
    description="Полная реализация всех workflow без урезаний",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
db_pool = None
redis_client = None
genai_client = None

# Pydantic модели
class SearchProductsRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    tenantId: str = Field(..., description="ID тенанта")
    cohereApiKey: str = Field(..., description="API ключ Cohere")
    sessionId: str = Field(..., description="ID сессии")
    limit: int = Field(10, description="Лимит результатов")
    occasion: Optional[str] = Field(None, description="Повод")
    min_price: Optional[int] = Field(None, description="Минимальная цена")
    max_price: Optional[int] = Field(None, description="Максимальная цена")

class ComplexQueryRequest(BaseModel):
    query_type: str = Field(..., description="Тип запроса")
    tenant_id: str = Field(..., description="ID тенанта")
    sessionId: str = Field(..., description="ID сессии")
    userPhone: str = Field(..., description="Телефон пользователя")
    params: Dict[str, Any] = Field(..., description="Параметры")
    input_products: Optional[List[Dict]] = Field(None, description="Входные товары")

# Утилиты для работы с базой данных
async def get_db_connection():
    """Получение пула соединений с базой данных"""
    global db_pool
    if db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
        
        try:
            db_pool = await asyncpg.create_pool(
                database_url, 
                min_size=5, 
                max_size=20,
                command_timeout=60
            )
            logger.info("Database pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")
    
    return db_pool

async def get_redis_client():
    """Получение Redis клиента"""
    global redis_client
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis client created successfully")
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            raise HTTPException(status_code=500, detail="Redis connection failed")
    
    return redis_client

async def initialize_gemini():
    """Инициализация Gemini API"""
    global genai_client
    if genai_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
        
        try:
            genai.configure(api_key=api_key)
            genai_client = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
                generation_config=genai.types.GenerationConfig(
                    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
                )
            )
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise HTTPException(status_code=500, detail="Gemini initialization failed")
    
    return genai_client

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_cohere_embedding(text: str, api_key: str = None) -> List[float]:
    """Получение эмбеддинга от Cohere с retry логикой"""
    if not api_key:
        api_key = os.getenv("COHERE_API_KEY")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="Cohere API key not configured")
    
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
                    "model": os.getenv("COHERE_MODEL", "embed-multilingual-v3.0"),
                    "input_type": "search_query"
                },
                timeout=int(os.getenv("COHERE_TIMEOUT", "30"))
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except httpx.HTTPError as e:
            logger.error(f"Cohere API error: {e}")
            raise HTTPException(status_code=500, detail=f"Cohere API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Cohere API: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

async def get_tenant_config(tenant_id: str) -> Optional[Dict]:
    """Получение конфигурации тенанта - ТОЧНАЯ КОПИЯ из configLoad"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        try:
            # Устанавливаем tenant для RLS - ТОЧНО как в workflow
            await conn.execute(f"SET app.tenant_id = '{tenant_id}';")
            
            # ТОЧНЫЙ SQL из workflow
            result = await conn.fetchrow("""
                SELECT persona_prompt, product_noun_singular, product_noun_plural, 
                       currency_char, currency_divisor
                FROM flbot.tenants 
                WHERE tenant_id = $1
            """, tenant_id)
            
            if result:
                return {
                    "persona_prompt": result["persona_prompt"],
                    "product_noun_singular": result["product_noun_singular"],
                    "product_noun_plural": result["product_noun_plural"],
                    "currency_char": result["currency_char"],
                    "currency_divisor": result["currency_divisor"]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting tenant config: {e}")
            return None

async def search_products_in_db(
    tenant_id: str,
    embedding: List[float],
    limit: int,
    occasion: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None
) -> List[Dict]:
    """Поиск товаров - ТОЧНАЯ КОПИЯ temporary_product_search функции"""
    
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        try:
            # Устанавливаем tenant для RLS
            await conn.execute(f"SET app.tenant_id = '{tenant_id}';")
            
            # ТОЧНАЯ КОПИЯ temporary_product_search функции из первого workflow
            temp_function_sql = """
            CREATE OR REPLACE FUNCTION temporary_product_search(
                p_tenant_id VARCHAR,
                p_occasion VARCHAR,
                p_min_price INT,
                p_max_price INT,
                p_limit INT,
                p_embedding VECTOR
            ) RETURNS JSONB AS $$
            DECLARE
                found_products jsonb;
                alternative_products jsonb;
            BEGIN
                -- Этап 1: Строгий поиск
                SELECT jsonb_agg(t) INTO found_products
                FROM (
                    SELECT name, description, image_url, 
                           (price_int / 100.0)::numeric(10, 2) AS price,
                           price_int, product_id
                    FROM flbot.products
                    WHERE
                      tenant_id = p_tenant_id AND stock = TRUE
                      AND (description ILIKE '%' || p_occasion || '%' OR p_occasion IS NULL)
                      AND (price_int >= p_min_price * 100 OR p_min_price IS NULL)
                      AND (price_int <= p_max_price * 100 OR p_max_price IS NULL)
                    ORDER BY embedding <=> p_embedding
                    LIMIT p_limit
                ) t;

                -- Этап 2: Альтернативный поиск
                IF found_products IS NULL THEN
                    SELECT jsonb_agg(t) INTO alternative_products
                    FROM (
                        SELECT name, description, image_url, 
                               (price_int / 100.0)::numeric(10, 2) AS price,
                               price_int, product_id, 'alternative' as type
                        FROM flbot.products
                        WHERE tenant_id = p_tenant_id AND stock = TRUE
                        ORDER BY embedding <=> p_embedding
                        LIMIT 3 -- Жесткий лимит для альтернатив
                    ) t;
                END IF;

                RETURN COALESCE(found_products, alternative_products, '[]'::jsonb);
            END;
            $$ LANGUAGE plpgsql;
            """
            
            await conn.execute(temp_function_sql)
            
            # Вызов функции с точными параметрами
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            result = await conn.fetchval(
                "SELECT temporary_product_search($1, $2, $3, $4, $5, $6::vector)",
                tenant_id,
                occasion if occasion else '',
                min_price,
                max_price,
                limit,
                embedding_str
            )
            
            return result or []
            
        except Exception as e:
            logger.error(f"Error in search_products_in_db: {e}")
            return []

async def execute_sql_query(tenant_id: str, query: str) -> Union[List[Dict], Dict]:
    """Выполнение SQL-запроса - ТОЧНАЯ ЗАМЕНА SQL Runner с обработкой ошибок"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        try:
            # Устанавливаем tenant для RLS
            await conn.execute(f"SET app.tenant_id = '{tenant_id}';")
            
            # Выполняем запрос
            result = await conn.fetch(query)
            
            # Возвращаем как список словарей - ТОЧНО как в SQL Runner
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            # Возвращаем ошибку в формате, который ожидает код
            return {"error": True, "message": str(e)}

def find_set_from_anchor(anchor: Dict, product_list: List[Dict], remaining_qty: int, remaining_budget: int) -> Optional[List[Dict]]:
    """ТОЧНАЯ КОПИЯ findSet функции из Set Generator"""
    product_set = [anchor]
    current_budget = remaining_budget
    excluded_ids = {anchor["product_id"]}
    
    for i in range(remaining_qty):
        remaining_slots = remaining_qty - i
        target_price = current_budget / remaining_slots if remaining_slots > 0 else 0
        
        best_fit = None
        min_diff = float('inf')
        
        for p in product_list:
            if p["product_id"] in excluded_ids:
                continue
            if p["price_int"] > current_budget:
                continue
                
            diff = abs(p["price_int"] - target_price)
            if diff < min_diff:
                min_diff = diff
                best_fit = p
        
        if best_fit:
            product_set.append(best_fit)
            current_budget -= best_fit["price_int"]
            excluded_ids.add(best_fit["product_id"])
        else:
            return None
    
    return product_set

def generate_product_sets(products: List[Dict], quantity: int, max_budget_int: int) -> List[Dict]:
    """ТОЧНАЯ КОПИЯ генерации сетов из Set Generator"""
    if not products or len(products) < quantity:
        return []
    
    generated_sets = []
    
    def add_unique_set(product_set):
        if not product_set:
            return
        set_id = tuple(sorted(p["product_id"] for p in product_set))
        if not any(s["id"] == set_id for s in generated_sets):
            generated_sets.append({"id": set_id, "items": product_set})
    
    # Стратегия 1: Сбалансированный якорь
    target_price = max_budget_int / quantity
    balanced_anchor = min(products, key=lambda p: abs(p["price_int"] - target_price))
    if balanced_anchor:
        balanced_set = find_set_from_anchor(
            balanced_anchor, 
            products, 
            quantity - 1, 
            max_budget_int - balanced_anchor["price_int"]
        )
        add_unique_set(balanced_set)
    
    # Стратегия 2: Самый дешевый якорь
    cheapest_anchor = min(products, key=lambda p: p["price_int"])
    if cheapest_anchor:
        budget_set = find_set_from_anchor(
            cheapest_anchor, 
            products, 
            quantity - 1,
            max_budget_int - cheapest_anchor["price_int"]
        )
        add_unique_set(budget_set)
    
    # Стратегия 3: Премиум якорь
    sorted_products = sorted(products, key=lambda p: p["price_int"], reverse=True)
    premium_anchor = next((p for p in sorted_products if p["price_int"] < max_budget_int * 1.1), None)
    if premium_anchor:
        premium_set = find_set_from_anchor(
            premium_anchor, 
            products, 
            quantity - 1,
            int(max_budget_int * 1.1) - premium_anchor["price_int"]
        )
        add_unique_set(premium_set)
    
    # Дополнительные попытки если меньше 3 сетов
    if len(generated_sets) < 3:
        for anchor in reversed(sorted_products):
            if len(generated_sets) >= 3:
                break
            additional_set = find_set_from_anchor(
                anchor, 
                products, 
                quantity - 1,
                int(max_budget_int * 1.1) - anchor["price_int"]
            )
            add_unique_set(additional_set)
    
    return generated_sets[:3]  # Максимум 3 набора

async def get_chat_history(session_id: str) -> List[Dict]:
    """Получение истории чата из Redis"""
    redis_client = await get_redis_client()
    
    try:
        history_key = f"chat_history:{session_id}"
        history_data = await redis_client.get(history_key)
        
        if history_data:
            return json.loads(history_data)
        return []
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

async def save_chat_history(session_id: str, user_message: str, ai_response: str):
    """Сохранение истории чата в Redis"""
    redis_client = await get_redis_client()
    
    try:
        history_key = f"chat_history:{session_id}"
        history = await get_chat_history(session_id)
        
        # Добавляем новые сообщения
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_response})
        
        # Ограничиваем историю последними 20 сообщениями
        if len(history) > 20:
            history = history[-20:]
        
        ttl = int(os.getenv("CHAT_HISTORY_TTL", "86400"))  # 24 часа
        await redis_client.setex(
            history_key, 
            ttl,
            json.dumps(history, ensure_ascii=False)
        )
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_gemini_with_memory(prompt: str, session_id: str) -> str:
    """Вызов Gemini с учетом контекста сессии - ТОЧНАЯ ЗАМЕНА miniOrator"""
    try:
        # Инициализация Gemini
        model = await initialize_gemini()
        
        # Получение истории чата
        chat_history = await get_chat_history(session_id)
        
        # Формирование контекста
        context_messages = []
        for msg in chat_history[-10:]:  # Последние 10 сообщений для контекста
            context_messages.append(f"{msg['role']}: {msg['content']}")
        
        # Формирование финального промпта
        full_prompt = prompt
        if context_messages:
            full_prompt = f"""
Контекст предыдущих сообщений:
{chr(10).join(context_messages)}

Текущий запрос:
{prompt}
"""
        
        # Генерация ответа
        response = await model.generate_content_async(full_prompt)
        
        if response and response.text:
            # Сохраняем в историю
            await save_chat_history(session_id, prompt, response.text)
            return response.text
        else:
            return "К сожалению, произошла ошибка при генерации ответа."
            
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        return "К сожалению, произошла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."

async def log_analytics_event(
    tenant_id: str,
    session_id: str,
    event_type: str,
    metadata: Dict[str, Any],
    user_phone: Optional[str] = None
):
    """Логирование события - ТОЧНАЯ КОПИЯ из обоих workflow"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        try:
            if user_phone:
                # Формат для сложных запросов - с event_timestamp, user_phone
                await conn.execute(
                    """
                    INSERT INTO analytics.events 
                    (event_timestamp, tenant_id, session_id, user_phone, event_type, metadata)
                    VALUES (NOW(), $1, $2, $3, $4, $5)
                    """,
                    tenant_id, session_id, user_phone, event_type, json.dumps(metadata)
                )
            else:
                # Формат для простых запросов - без event_timestamp, user_phone
                await conn.execute(
                    """
                    INSERT INTO analytics.events 
                    (tenant_id, session_id, event_type, metadata)
                    VALUES ($1, $2, $3, $4)
                    """,
                    tenant_id, session_id, event_type, json.dumps(metadata)
                )
        except Exception as e:
            logger.error(f"Error logging analytics event: {e}")

# API Endpoints
@app.post("/search-products")
async def search_products(request: SearchProductsRequest):
    """Поиск товаров - ПОЛНАЯ РЕАЛИЗАЦИЯ первого workflow"""
    try:
        # Получаем эмбеддинг - ТОЧНО как в Get Embedding узле
        embedding = await get_cohere_embedding(request.query, request.cohereApiKey)
        
        # Поиск в базе данных - ТОЧНО как в Execute a SQL query узле
        products = await search_products_in_db(
            tenant_id=request.tenantId,
            embedding=embedding,
            limit=request.limit,
            occasion=request.occasion,
            min_price=request.min_price,
            max_price=request.max_price
        )
        
        # Обработка результатов - ТОЧНО как в Code узле
        if not products or len(products) == 0:
            final_result = "К сожалению, по вашему запросу ничего не найдено."
        else:
            final_result = products
        
        # Логирование - ТОЧНО как в Tool Usage узле
        await log_analytics_event(
            tenant_id=request.tenantId,
            session_id=request.sessionId,
            event_type="TOOL_USED",
            metadata={
                "tool_name": "find_relevant_products",
                "query": request.query,
                "min_price": request.min_price,
                "max_price": request.max_price,
                "occasion": request.occasion,
                "results_count": len(products) if isinstance(products, list) else 0,
                "success": isinstance(products, list) and len(products) > 0
            }
        )
        
        return {"result": final_result}
        
    except Exception as e:
        logger.error(f"Error in search_products: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при поиске товаров")

@app.post("/analyze-complex-query")
async def analyze_complex_query(request: ComplexQueryRequest):
    """Анализ сложных запросов - ПОЛНАЯ РЕАЛИЗАЦИЯ второго workflow"""
    try:
        # Получаем конфигурацию тенанта - ТОЧНО как в configLoad
        config = await get_tenant_config(request.tenant_id)
        if not config:
            raise HTTPException(status_code=404, detail="Tenant configuration not found")
        
        # Объединяем данные - ТОЧНО как в Merge узле
        merged_data = {
            "query_type": request.query_type,
            "params": request.params,
            "input_products": request.input_products,
            "tenant_id": request.tenant_id,
            "sessionId": request.sessionId,
            "userPhone": request.userPhone,
            "persona_prompt": config["persona_prompt"],
            "product_noun_singular": config["product_noun_singular"],
            "product_noun_plural": config["product_noun_plural"],
            "currency_char": config["currency_char"],
            "currency_divisor": config["currency_divisor"]
        }
        
        # Data Preparation - ТОЧНАЯ КОПИЯ логики из Data Preparation узла
        trigger_data = {
            "query_type": merged_data["query_type"],
            "params": merged_data["params"],
            "input_products": merged_data["input_products"],
            "tenant_id": merged_data["tenant_id"],
            "sessionId": merged_data["sessionId"],
            "userPhone": merged_data["userPhone"]
        }
        
        config_data = {
            "persona_prompt": merged_data["persona_prompt"],
            "product_noun_singular": merged_data["product_noun_singular"],
            "product_noun_plural": merged_data["product_noun_plural"],
            "currency_char": merged_data["currency_char"],
            "currency_divisor": merged_data["currency_divisor"]
        }
        
        result = {
            "products": None,
            "instruction": "",
            "config": config_data,
            "params": trigger_data["params"] or {},
            "query_type": trigger_data["query_type"]
        }
        
        # Обработка BUDGET_SPLIT - ТОЧНАЯ КОПИЯ из Data Preparation
        if result["query_type"] == "BUDGET_SPLIT":
            quantity = result["params"].get("quantity")
            max_budget = result["params"].get("max_budget")
            
            if not isinstance(quantity, (int, float)) or quantity <= 0 or not isinstance(max_budget, (int, float)) or max_budget <= 0:
                result["instruction"] = "Кажется, в запросе были неверные параметры для количества или бюджета."
            else:
                max_budget_int = int(max_budget * config_data["currency_divisor"])
                
                if trigger_data["input_products"] and isinstance(trigger_data["input_products"], list) and len(trigger_data["input_products"]) > 0:
                    result["products"] = trigger_data["input_products"]
                else:
                    # Проверка самых дешевых товаров
                    cheapest_query = f"SELECT price_int FROM flbot.products WHERE tenant_id = '{trigger_data['tenant_id']}' AND stock = TRUE ORDER BY price_int ASC LIMIT {quantity};"
                    cheapest_products = await execute_sql_query(trigger_data["tenant_id"], cheapest_query)
                    
                    if cheapest_products and not isinstance(cheapest_products, dict) and len(cheapest_products) >= quantity:
                        min_possible_total = sum(p["price_int"] for p in cheapest_products)
                        
                        if min_possible_total <= max_budget_int * 1.1:
                            price_limit = int(max_budget_int * 1.1) - sum(p["price_int"] for p in cheapest_products[:quantity-1])
                            all_products_query = f"SELECT product_id, name, price_int FROM flbot.products WHERE tenant_id = '{trigger_data['tenant_id']}' AND stock = TRUE AND price_int <= {price_limit} ORDER BY price_int ASC;"
                            result["products"] = await execute_sql_query(trigger_data["tenant_id"], all_products_query)
                        else:
                            min_total_formatted = f"{min_possible_total / config_data['currency_divisor']:.2f}"
                            result["instruction"] = f"Подобрать {quantity} {config_data['product_noun_plural']} на бюджет до {max_budget} {config_data['currency_char']} будет сложно."
                    elif cheapest_products and isinstance(cheapest_products, dict) and cheapest_products.get("error"):
                        result["instruction"] = "К сожалению, при поиске произошла техническая ошибка."
                    else:
                        result["instruction"] = f"К сожалению, у нас в наличии меньше {quantity} {config_data['product_noun_plural']}."
        
        # Обработка FIND_EXTREME_PRICE - ТОЧНАЯ КОПИЯ из Data Preparation
        elif result["query_type"] == "FIND_EXTREME_PRICE":
            sort_order = result["params"].get("sort_order", "ASC")
            query = f"SELECT product_id, name, description, image_url, price_int FROM flbot.products WHERE tenant_id = '{trigger_data['tenant_id']}' AND stock = TRUE ORDER BY price_int {sort_order} LIMIT 1;"
            result["products"] = await execute_sql_query(trigger_data["tenant_id"], query)
        
        # Set Generator - ТОЧНАЯ КОПИЯ логики из Set Generator узла
        products = result["products"]
        initial_instruction = result["instruction"]
        config = result["config"]
        params = result["params"]
        query_type = result["query_type"]
        
        instruction = initial_instruction
        final_results = []
        
        if instruction:
            final_results = [{"error": instruction}]
        elif query_type == "BUDGET_SPLIT":
            quantity = params.get("quantity")
            max_budget_int = int(params.get("max_budget", 0) * config["currency_divisor"])
            
            if not products or not isinstance(products, list) or len(products) < quantity:
                instruction = f"К сожалению, у нас недостаточно подходящих {config['product_noun_plural']}."
            else:
                # Генерация сетов
                generated_sets = generate_product_sets(products, quantity, max_budget_int)
                
                if len(generated_sets) > 0:
                    final_results = []
                    for s in generated_sets:
                        total = sum(p["price_int"] for p in s["items"])
                        final_results.append({
                            "total_price": f"{total / config['currency_divisor']:.2f}",
                            "items": [
                                {
                                    "name": p["name"],
                                    "price": f"{p['price_int'] / config['currency_divisor']:.2f}"
                                } for p in s["items"]
                            ]
                        })
                    
                    # ТОЧНАЯ КОПИЯ инструкции с правилами форматирования
                    instruction = f"""Отличная идея! Мы подобрали для вас несколько готовых комбинаций из {quantity} {config['product_noun_plural']} на бюджет до {params.get('max_budget')} {config['currency_char']}. Посмотрите, какой вариант вам нравится больше.

### ПРАВИЛА ФОРМАТИРОВАНИЯ:
- Для каждого сета укажи его номер или название (например, "Комбинация 1").
- Внутри каждого сета перечисли все товары из поля "items" в виде списка.
- Для каждого товара в списке ОБЯЗАТЕЛЬНО укажи его название и цену из полей "name" и "price".
- После списка товаров в сете ОБЯЗАТЕЛЬНО укажи общую стоимость из поля "total_price".
- Если общая стоимость превышает бюджет, сделай на этом акцент, например: "(немного выше бюджета)".

### ПРИМЕР ИДЕАЛЬНОГО ФОРМАТА:
*   *Комбинация 1: "Сбалансированная"*
    *   Букет "Нежность" – *25.00 AZN*
    *   Букет "Радость" – *30.00 AZN*
    *   Букет "Улыбка" – *40.00 AZN*
    *   *Общая стоимость: 95.00 AZN*"""
                else:
                    instruction = "Мы постарались, но, к сожалению, не смогли составить подходящие комбинации."
        
        elif query_type == "FIND_EXTREME_PRICE":
            if not products or not isinstance(products, list) or len(products) == 0:
                instruction = "Вежливо сообщи пользователю, что наш каталог пуст."
            else:
                product = products[0]
                final_results = [product]
                price_type = "самый дорогой" if params.get("sort_order") == "DESC" else "самый доступный"
                instruction = f"Пользователь спросил про наш {price_type} {config['product_noun_singular']}. Вот он. Опиши его красиво, используя его название, описание и цену."
        
        if not instruction:
            instruction = "Произошла непредвиденная ошибка."
            final_results = [{"error": "Internal instruction generation failed"}]
        
        # Генерация промпта для оратора - ТОЧНАЯ КОПИЯ из Set Generator
        prompt_for_orator = f"""{config['persona_prompt']}

### ЗАДАЧА
{instruction}

### ДАННЫЕ
{json.dumps(final_results, ensure_ascii=False)}"""
        
        # Вызов miniOrator - ТОЧНАЯ ЗАМЕНА узла miniOrator
        ai_response = await call_gemini_with_memory(prompt_for_orator, request.sessionId)
        
        # Обработка ответа - ТОЧНАЯ КОПИЯ из formattingResponse
        if not ai_response or not isinstance(ai_response, str) or ai_response.strip() == "":
            final_ai_response = "К сожалению, произошла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."
        else:
            final_ai_response = ai_response
        
        # Логирование - ТОЧНАЯ КОПИЯ из Log: Complex Query Analyzed
        await log_analytics_event(
            tenant_id=request.tenant_id,
            session_id=request.sessionId,
            event_type="COMPLEX_QUERY_ANALYZED",
            metadata={
                "tool_name": "analyze_complex_query",
                "query_type": request.query_type,
                "params": request.params,
                "results_count": len(final_results),
                "success": len(final_results) > 0 and not any(r.get("error") for r in final_results),
                "error_message": final_results[0].get("error") if final_results and final_results[0].get("error") else ""
            },
            user_phone=request.userPhone
        )
        
        return {
            "result": final_ai_response,
            "prompt_for_orator": prompt_for_orator,
            "final_results": final_results
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_complex_query: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при анализе сложного запроса")

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "service": "unified-product-mcp",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0"
    }

# Обработчики запуска и остановки
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("Запуск полного MCP сервера...")
    
    # Инициализация всех соединений
    await get_db_connection()
    await get_redis_client()
    await initialize_gemini()
    
    logger.info("MCP сервер с полной логикой успешно запущен")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    global db_pool, redis_client
    
    if db_pool:
        await db_pool.close()
    
    if redis_client:
        await redis_client.close()
    
    logger.info("MCP сервер остановлен")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
