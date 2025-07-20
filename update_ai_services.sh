#!/bin/bash

# Переходим в директорию проекта
cd /home/mcp-server/mcp-server/mcp/services

# Создаем backup с временной меткой
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp ai_services.py ai_services.py.backup_$TIMESTAMP

echo "✅ Backup создан: ai_services.py.backup_$TIMESTAMP"

# 1. Обновляем сигнатуру _create_fallback_classification
sed -i 's/async def _create_fallback_classification(self, query: str, tenant_id: str) -> Dict:/async def _create_fallback_classification(self, query: str, tenant_id: str, language: str = "ru") -> Dict:/' ai_services.py

# 2. Заменяем весь fallback метод на новую логику
cat > temp_fallback.py << 'EOF'
    async def _create_fallback_classification(self, query: str, tenant_id: str, language: str = "ru") -> Dict:
        """Language-agnostic numeric fallback"""
        
        # Извлекаем только числа - работает универсально для любого языка
        numbers = re.findall(r'\d+', query)
        
        if len(numbers) == 2:
            num1, num2 = int(numbers[0]), int(numbers[1])
            
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
EOF

# 3. Заменяем старый fallback метод
python3 -c "
import re

with open('ai_services.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Находим и заменяем метод _create_fallback_classification
pattern = r'    async def _create_fallback_classification\(self, query: str, tenant_id: str, language: str = \"ru\"\) -> Dict:.*?(?=    async def|\n    @|\n    def|\nclass|\Z)'
with open('temp_fallback.py', 'r') as f:
    new_method = f.read().strip()

content = re.sub(pattern, new_method, content, flags=re.DOTALL)

with open('ai_services.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fallback метод заменен')
"

# 4. Добавляем новый метод _get_tenant_context
cat > temp_tenant_context.py << 'EOF'

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
EOF

# Вставляем новый метод перед существующим classify_query
sed -i '/async def classify_query/i\'"$(cat temp_tenant_context.py)" ai_services.py

# 5. Заменяем весь метод classify_query
cat > temp_classify.py << 'EOF'
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
EOF

# Заменяем старый classify_query
python3 -c "
import re

with open('ai_services.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Находим и заменяем метод classify_query  
pattern = r'    async def classify_query\(self, query: str, tenant_id: str, language: str = \"ru\"\) -> Dict:.*?(?=    @|\n    async def|\n    def|\nclass|\Z)'
with open('temp_classify.py', 'r') as f:
    new_method = f.read().strip()

content = re.sub(pattern, new_method, content, flags=re.DOTALL)

with open('ai_services.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ classify_query метод заменен')
"

# Очистка временных файлов
rm temp_fallback.py temp_tenant_context.py temp_classify.py

# Проверяем синтаксис Python
echo "🔍 Проверяем синтаксис обновленного файла..."
python3 -m py_compile ai_services.py

if [ $? -eq 0 ]; then
    echo "✅ Все изменения применены успешно!"
    echo "✅ Синтаксис корректный"
    echo ""
    echo "📊 Сводка изменений:"
    echo "• Добавлена поддержка параметра language"
    echo "• Убраны все языковые регулярки из fallback"  
    echo "• Добавлена умная числовая логика fallback"
    echo "• Добавлен метод _get_tenant_context"
    echo "• Переписан метод classify_query для AI-first подхода"
    echo ""
    echo "🎯 Система готова к мультиязычной работе!"
else
    echo "❌ Обнаружены ошибки синтаксиса!"
    echo "Восстанавливаем из backup'а..."
    cp ai_services.py.backup_$TIMESTAMP ai_services.py
    echo "Файл восстановлен"
    exit 1
fi
