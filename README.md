# data-anonymizer
Подключаемый модуль для обезличивания табличных данных перед обучением моделей машинного обучения.
Содержит два класса: DifferentialPrivacyProcessor - для осуществления обезличивания без обратимости при помощи дифференциальной приватности
и AnonimyzerWithIDs для скрытия информации при помощи введния идентификаторов и изменения семантики путем представления данных в вещественном формате от 0 до 1 

В файле samples.ipynb содержаться примеры использования модуля для обучения моделей машинного обучения для задачи регрессии

### Руководство пользователя для модуля обработки данных

#### Описание модуля
Включены такие функции, как добавление шума к числовым данным, рандомизация категориальных данных, анонимизация имен, электронных адресов и адресов проживания. Также доступен механизм введения идентификаторов колонок и значений.

---

#### Основные возможности:
1. **Дифференциальная приватность**:
   - Добавление шума к числовым данным с использованием Лапласового механизма.
   - Рандомизация категориальных данных.
   
2. **Анонимизация**:
   - Генерация случайных электронных адресов.
   - Анонимизация имен через перемешивание фамилий, имен и отчеств между строками.
   - Извлечение городов из полного адреса для скрытия специфики адресов.

3. **Работа с идентификаторами**:
   - Переименование колонок в формат `x1`, `x2`, ...
   - Нормализация числовых данных с сохранением таблицы соответствий.

---

### Инструкция по использованию

#### Инициализация

```python
from anonymizer import DifferentialPrivacyProcessor, AnonimyzerWithIDs

# Инициализация процессоров
dp_processor = DifferentialPrivacyProcessor(
    epsilon_numeric=1.0, 
    epsilon_categorical=1.0, 
    keep_domain=True
)

anonymizer = AnonimyzerWithIDs(
    anonymize_columns=True,
    models_return=True
)
```

---

#### Обработка данных с помощью `DifferentialPrivacyProcessor`

##### 1. **Добавление шума к числовым данным**

- **Методы расчета чувствительности**:
  - `"range"`: Максимальный размах значений $$\(max - min\)$$.
  - `"std"`: Стандартное отклонение $$\(\sigma\)$$.
  - Пользовательское значение.

Пример:
```python
data['numeric_column'] = dp_processor.add_numeric_noise(data['numeric_column'], method="range")
```

- **Рекомендации по выбору \(\varepsilon\)**:
  - $$\(0.1 \leq \varepsilon \leq 1.0\)$$: Высокий уровень приватности, больше шума.
  - $$\(\varepsilon > 1.0\)$$: Низкий уровень приватности, меньше шума.

##### 2. **Рандомизация категориальных данных**

Рандомизация изменяет категории с учетом вероятности, зависящей от $$\(\varepsilon\)$$. Подходит для случаев с ограниченным числом уникальных категорий.

Пример:
```python
data['category_column'] = dp_processor.randomize_categories(data['category_column'])
```

- **Рекомендации по выбору $$\(\varepsilon\)$$**:
  - Меньшее значение $$\(\varepsilon\)$$ увеличивает вероятность замены категории.

##### 3. **Анонимизация ФИО**

Перемешивание фамилий, имен и отчеств между строками:

```python
data['name_column'] = dp_processor.anonymize_names(data['name_column'])
```

##### 4. **Генерация синтетических электронных адресов**

```python
data['email_column'] = dp_processor.randomize_email(data['email_column'])
```

---

#### Обработка данных с помощью `AnonimyzerWithIDs`

##### 1. **Преобразование данных**

Обработка включает нормализацию числовых колонок, кодирование категорий и анонимизацию названий колонок:

```python
processed_data = anonymizer.preprocess_dataset(data, categorical_cols=['col1'], numeric_cols=['col2'])
```

##### 2. **Обратное преобразование**

Восстановление оригинальных данных из обработанного формата:
```python
original_data = anonymizer.reverse_prepocess(processed_data)
```

##### 3. **Рекомендации**:
- Укажите `models_return=True`, чтобы сохранить обученные модели кодировщиков и нормализаторов.
- Убедитесь, что при обратном преобразовании используются сохраненные модели.

---

### Вывод

Модуль поддерживает гибкую настройку параметров, таких как $$\(\varepsilon\)$$, методы расчета чувствительности и другие. Он подходит для обработки различных типов данных, обеспечивая баланс между приватностью и полезностью данных.
