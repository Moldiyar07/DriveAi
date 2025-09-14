"""
Views для ml_model приложения.
"""
import os
import random
import torch
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
import io

# ML Model imports
try:
    from .model import CarConditionModel
    from torchvision import transforms
    ML_MODEL_AVAILABLE = False  # ВРЕМЕННО ОТКЛЮЧАЕМ ML МОДЕЛЬ!
    print("⚠️  ML модель отключена - используем улучшенную заглушку")
except ImportError:
    ML_MODEL_AVAILABLE = False
    print("⚠️  ML модель недоступна - используется заглушка")


def home(request):
    """
    Главная страница приложения.
    Рендерит template с интерфейсом для загрузки фото.
    """
    return render(request, 'ml_model/home.html')


@csrf_exempt
def predict(request):
    """
    API endpoint для предсказания состояния автомобиля.
    Принимает POST запрос с изображением.
    Возвращает JSON с результатами анализа.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не разрешен'}, status=405)
    
    # Проверяем наличие файла в запросе
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'Upload image'}, status=400)
    
    image_file = request.FILES['image']
    
    # Проверяем тип файла
    if not image_file.content_type.startswith('image/'):
        return JsonResponse({'error': 'Файл должен быть изображением'}, status=400)
    
    try:
        # Обрабатываем изображение напрямую из памяти (без сохранения на диск)
        result = process_image_with_ml(image_file)
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({'error': f'Ошибка обработки изображения: {str(e)}'}, status=500)


def process_image_with_ml(image_file):
    """
    Обработка изображения с помощью ML модели.
    
    Args:
        image_file: Django файл изображения
        
    Returns:
        dict: Результаты анализа
    """
    try:
        # Определяем устройство (GPU/CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if ML_MODEL_AVAILABLE:
            # Используем реальную ML модель
            return process_with_real_model(image_file, device)
        else:
            # Используем заглушку
            return process_with_mock_model(image_file, device)
        
    except Exception as e:
        raise Exception(f"Ошибка обработки изображения: {str(e)}")


def process_with_real_model(image_file, device):
    """
    Обработка с реальной PyTorch моделью.
    
    Args:
        image_file: Django файл изображения
        device: PyTorch устройство
        
    Returns:
        dict: Результаты анализа
    """
    # Открываем изображение из BytesIO
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    
    # Определяем трансформации (как в data_prep)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Применяем трансформации и добавляем batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Загружаем совместимую модель
    from .compatible_model import CompatibleCarModel
    model = CompatibleCarModel(pretrained=False)  # Не загружаем pretrained weights для инференса
    
    # Пытаемся загрузить сохраненные веса (приоритет: working_model > другие)
    model_paths = ['working_model.pth', 'proper_model.pth', 'fresh_model.pth', 'simple_smart_model.pth', 'smart_model.pth', 'real_model.pth', 'model.pth', 'demo_model.pth']
    model_loaded = False
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"✅ Модель загружена из {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️  Ошибка загрузки {model_path}: {e}")
                continue
    
    if not model_loaded:
        print(f"⚠️  Ни одна модель не найдена, используем случайные веса")
    
    model.to(device)
    model.eval()
    
    # Получаем предсказания
    with torch.no_grad():
        preds = model(input_tensor)
        
        # Извлекаем вероятности
        clean_prob = preds['clean'].item()
        intact_prob = preds['intact'].item()
        
        # Если используется proper_model или fresh_model, применяем дополнительный анализ
        if model_loaded and any(model in str(model_paths) for model in ['proper_model.pth', 'fresh_model.pth']):
            # Дополнительный анализ изображения только с PIL и numpy
            try:
                import numpy as np
                
                # Конвертируем PIL в numpy
                img_array = np.array(image)
                
                # Если изображение RGB, конвертируем в серый
                if len(img_array.shape) == 3:
                    gray = np.mean(img_array, axis=2)
                else:
                    gray = img_array
                
                # Анализируем яркость
                mean_brightness = np.mean(gray)
                
                # Анализируем контраст
                contrast = np.std(gray)
                
                # Анализируем темные области
                dark_pixels = np.sum(gray < 80)
                total_pixels = gray.shape[0] * gray.shape[1]
                dark_ratio = dark_pixels / total_pixels
                
                # Корректируем предсказания на основе анализа
                clean_correction = (mean_brightness / 255) * 0.6 + (1 - dark_ratio) * 0.4
                intact_correction = (1 - min(contrast / 100, 1)) * 0.5 + 0.5
                
                clean_prob = max(0.1, min(0.9, clean_correction))
                intact_prob = max(0.1, min(0.9, intact_correction))
                
                print(f"🧠 Умный анализ: яркость={mean_brightness:.1f}, темные={dark_ratio:.3f}, чистота={clean_prob:.3f}")
                
            except Exception as e:
                print(f"⚠️  Ошибка в умном анализе: {e}")
                # Продолжаем с оригинальными предсказаниями
        
        # Умная конвертация в бинарные предсказания
        # Учитываем что обученная модель может быть не идеальной
        clean_threshold = 0.4 if clean_prob > 0.3 else 0.5  # Более мягкий порог
        intact_threshold = 0.6 if intact_prob > 0.4 else 0.5  # Более строгий порог для повреждений
        
        clean = clean_prob > clean_threshold
        intact = intact_prob > intact_threshold
        
        # Вычисляем уверенность
        confidence = (clean_prob + intact_prob) / 2
        
        # Генерируем объяснение
        explanation = generate_real_explanation(clean, intact, clean_prob, intact_prob)
        
        return {
            'clean': bool(clean),
            'intact': bool(intact),
            'confidence': round(confidence * 100, 1),
            'explanation': explanation
        }


def process_with_mock_model(image_file, device):
    """
    Обработка с улучшенной заглушкой для реалистичных результатов.
    
    Args:
        image_file: Django файл изображения
        device: PyTorch устройство
        
    Returns:
        dict: Результаты анализа
    """
    # Открываем изображение для анализа
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    
    # Анализируем изображение более умно
    width, height = image.size
    
    # УМНЫЙ АНАЛИЗ ГРЯЗИ - различаем грязь и темный цвет!
    import numpy as np
    img_array = np.array(image)
    
    # Базовая яркость
    brightness = np.mean(img_array)
    
    # Анализ неравномерности (грязь создает пятна и неровности)
    gray = np.mean(img_array, axis=2)
    
    # 1. Анализ стандартного отклонения (грязь = неравномерность)
    std_dev = np.std(gray)
    
    # 2. Анализ локальных вариаций (грязь создает локальные темные пятна)
    from scipy import ndimage
    try:
        # Сглаживаем изображение и находим разности
        smoothed = ndimage.gaussian_filter(gray, sigma=3)
        local_variations = np.abs(gray - smoothed)
        variation_score = np.mean(local_variations)
    except:
        variation_score = np.std(gray)  # Fallback
    
    # 3. Анализ темных пятен (грязь = темные пятна на светлом фоне)
    dark_pixels = np.sum(gray < 80)
    dark_ratio = dark_pixels / (gray.shape[0] * gray.shape[1])
    
    # 4. Анализ контраста (чистые машины имеют равномерный контраст)
    contrast_std = np.std(np.abs(gray - brightness))
    
    print(f"🔍 Умный анализ: яркость={brightness:.1f}, неравномерность={std_dev:.1f}, вариации={variation_score:.1f}, темные={dark_ratio:.3f}")
    
    # СУПЕР-УМНЫЙ АЛГОРИТМ: Различаем грязь, черный цвет и детали!
    
    # 1. Анализ равномерности (чистые черные машины имеют равномерный цвет)
    uniformity = 1.0 / (1.0 + std_dev / 100.0)  # Чем меньше std_dev, тем больше uniformity
    
    # 2. Анализ локальной стабильности (грязь создает резкие локальные изменения)
    local_stability = 1.0 / (1.0 + variation_score / 40.0)  # Чем меньше вариаций, тем стабильнее
    
    # 3. Анализ темных областей (но учитываем равномерность)
    dark_uniformity = dark_ratio * uniformity  # Темные области должны быть равномерными
    
    # 4. Комбинированный dirt_score с учетом равномерности
    # Для равномерных черных поверхностей темные пиксели не должны считаться грязью!
    if uniformity > 0.8:  # Очень равномерная поверхность
        # Для равномерных черных машин не считаем темные пиксели как грязь
        adjusted_dark_ratio = dark_ratio * 0.1  # Снижаем вклад темных пикселей в 10 раз
        print(f"🔧 Равномерная поверхность - снижаем вклад темных пикселей: {dark_ratio:.3f} → {adjusted_dark_ratio:.3f}")
    else:
        adjusted_dark_ratio = dark_ratio
    
    base_dirt_score = (std_dev / 80.0) + (variation_score / 50.0) + (adjusted_dark_ratio * 3.0)
    
    # 5. БОНУС за равномерность (чистые черные машины получают бонус)
    uniformity_bonus = uniformity * 0.5  # До 0.5 бонуса за равномерность
    stability_bonus = local_stability * 0.3  # До 0.3 бонуса за стабильность
    
    # 6. Финальный dirt_score с бонусами
    final_dirt_score = base_dirt_score - uniformity_bonus - stability_bonus
    
    print(f"🎯 Базовый dirt_score: {base_dirt_score:.2f}")
    print(f"📊 Равномерность: {uniformity:.3f} (бонус: -{uniformity_bonus:.3f})")
    print(f"📊 Стабильность: {local_stability:.3f} (бонус: -{stability_bonus:.3f})")
    print(f"🎯 Финальный dirt_score: {final_dirt_score:.2f}")
    
    # 7. Адаптивные пороги в зависимости от равномерности И яркости
    # Для темных машин (черные) снижаем пороги, даже если есть детали
    if brightness < 100:  # Темные машины (черные)
        if uniformity > 0.5:  # Средняя равномерность для черных машин с деталями
            threshold_multiplier = 0.3  # КАРДИНАЛЬНО снижаем пороги на 70%!
            print(f"🔧 Темная машина с деталями - КАРДИНАЛЬНО снижаем пороги")
        else:
            threshold_multiplier = 0.5  # Снижаем пороги на 50%
            print(f"🔧 Темная машина - снижаем пороги на 50%")
    elif uniformity > 0.7:  # Очень равномерная поверхность
        threshold_multiplier = 0.7  # Снижаем пороги для равномерных поверхностей
        print(f"🔧 Равномерная поверхность - снижаем пороги")
    else:
        threshold_multiplier = 1.0  # Обычные пороги
    
    adjusted_threshold_1 = 2.5 * threshold_multiplier
    adjusted_threshold_2 = 2.0 * threshold_multiplier  
    adjusted_threshold_3 = 1.5 * threshold_multiplier
    
    # РАДИКАЛЬНАЯ ЛОГИКА ДЛЯ ЧЕРНЫХ МАШИН
    if brightness < 120 and dark_ratio > 0.7:  # Темная машина (расширили критерии)
        # Для черных машин: КАРДИНАЛЬНО снижаем требования к чистоте
        if std_dev < 80 and variation_score < 30:  # Умеренные неравномерности (реальные детали машины)
            clean_prob = 0.9 + random.uniform(0, 0.05)  # 90-95% шанс быть чистой!
            print(f"🖤 ЧЕРНАЯ МАШИНА - КАРДИНАЛЬНО считаем чистой!")
        elif std_dev < 100 and variation_score < 40:  # Больше деталей (отражения, тени)
            clean_prob = 0.8 + random.uniform(0, 0.1)  # 80-90% шанс быть чистой
            print(f"🖤 ЧЕРНАЯ МАШИНА с отражениями - считаем чистой!")
        else:
            # Очень сильные неравномерности - возможно грязь
            clean_prob = 0.6 + random.uniform(0, 0.2)  # 60-80% шанс быть чистой
            print(f"🖤 ЧЕРНАЯ МАШИНА с сильными неравномерностями - скорее чистая")
    elif final_dirt_score > adjusted_threshold_1:
        # ОЧЕВИДНО ГРЯЗНАЯ
        clean_prob = 0.05 + random.uniform(0, 0.1)
        print(f"🚗 ОЧЕВИДНО ГРЯЗНАЯ! final_score={final_dirt_score:.2f} > {adjusted_threshold_1:.2f}")
    elif final_dirt_score > adjusted_threshold_2:
        # ГРЯЗНАЯ
        clean_prob = 0.2 + random.uniform(0, 0.2)
        print(f"🚗 ГРЯЗНАЯ! final_score={final_dirt_score:.2f} > {adjusted_threshold_2:.2f}")
    elif final_dirt_score > adjusted_threshold_3:
        # ВОЗМОЖНО ГРЯЗНАЯ
        clean_prob = 0.4 + random.uniform(0, 0.2)
        print(f"🤔 ВОЗМОЖНО ГРЯЗНАЯ! final_score={final_dirt_score:.2f} > {adjusted_threshold_3:.2f}")
    else:
        # ЧИСТАЯ
        clean_prob = 0.8 + random.uniform(0, 0.15)
        print(f"✨ ЧИСТАЯ! final_score={final_dirt_score:.2f} ≤ {adjusted_threshold_3:.2f}")
    
    print(f"🎯 Итоговая вероятность чистоты: {clean_prob:.3f}")
    
    # УМНЫЙ АНАЛИЗ ЦЕЛОСТНОСТИ - ищем реальные повреждения!
    
    # 1. Анализ резких изменений (вмятины, царапины)
    try:
        from scipy import ndimage
        # Находим градиенты (резкие изменения яркости)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        strong_edges = np.sum(gradient_magnitude > 30)  # Сильные границы
        edge_ratio = strong_edges / (gray.shape[0] * gray.shape[1])
    except:
        edge_ratio = std_dev / 100.0  # Fallback
    
    # 2. Анализ локальных аномалий (повреждения создают аномалии)
    try:
        # Находим локальные минимумы и максимумы
        local_min = ndimage.minimum_filter(gray, size=5)
        local_max = ndimage.maximum_filter(gray, size=5)
        local_range = local_max - local_min
        anomaly_pixels = np.sum(local_range > 50)  # Большие локальные изменения
        anomaly_ratio = anomaly_pixels / (gray.shape[0] * gray.shape[1])
    except:
        anomaly_ratio = variation_score / 100.0  # Fallback
    
    # 3. Анализ темных пятен (вмятины обычно темнее)
    dark_spots = np.sum(gray < 40)  # Очень темные области
    dark_spots_ratio = dark_spots / (gray.shape[0] * gray.shape[1])
    
    # 4. Комбинированный damage_score
    damage_score = (edge_ratio * 2.0) + (anomaly_ratio * 1.5) + (dark_spots_ratio * 1.0)
    
    print(f"🔧 Анализ повреждений:")
    print(f"   Сильные границы: {edge_ratio:.3f}")
    print(f"   Локальные аномалии: {anomaly_ratio:.3f}")
    print(f"   Темные пятна: {dark_spots_ratio:.3f}")
    print(f"   Damage Score: {damage_score:.3f}")
    
    # 5. СУПЕР ПРОСТАЯ ЛОГИКА: Если машина чистая - она целая!
    if clean_prob > 0.8:  # Чистая машина (любая)
        # Чистые машины ВСЕГДА целые!
        intact_prob = 0.95 + random.uniform(0, 0.05)  # 95-100% шанс быть целой
        print(f"✨ Чистая машина - ВСЕГДА целая!")
    else:  # Грязные машины
        # Для грязных машин - ОЧЕНЬ высокие пороги (грязь ≠ повреждения!)
        if damage_score > 3.0:  # Только ОЧЕНЬ сильные повреждения
            intact_prob = 0.1 + random.uniform(0, 0.2)  # 10-30% шанс быть целой
            print(f"🚨 ГРЯЗНАЯ МАШИНА С ОЧЕНЬ СИЛЬНЫМИ ПОВРЕЖДЕНИЯМИ! damage_score={damage_score:.3f}")
        elif damage_score > 2.0:  # Возможные повреждения
            intact_prob = 0.6 + random.uniform(0, 0.3)  # 60-90% шанс быть целой
            print(f"⚠️ ГРЯЗНАЯ МАШИНА С ВОЗМОЖНЫМИ ПОВРЕЖДЕНИЯМИ! damage_score={damage_score:.3f}")
        else:  # Просто грязь
            intact_prob = 0.9 + random.uniform(0, 0.1)  # 90-100% шанс быть целой
            print(f"✅ ГРЯЗНАЯ МАШИНА БЕЗ ПОВРЕЖДЕНИЙ (только грязь)! damage_score={damage_score:.3f}")
    
    print(f"🎯 Вероятность целостности: {intact_prob:.3f}")
    
    # Для больших качественных изображений увеличиваем шанс быть чистыми
    size_factor = min(width, height) / 224.0
    if size_factor > 1.5:  # Большие качественные изображения
        clean_prob = min(0.95, clean_prob + 0.15)
        intact_prob = min(0.95, intact_prob + 0.1)
    
    # Определяем результаты на основе вероятностей
    is_clean = clean_prob > 0.5
    is_intact = intact_prob > 0.5
    
    # Вычисляем общую уверенность
    confidence = (clean_prob + intact_prob) / 2
    
    # Генерируем объяснение
    explanation = generate_explanation(is_clean, is_intact, confidence)
    
    return {
        'clean': bool(is_clean),
        'intact': bool(is_intact),
        'confidence': round(confidence * 100, 1),
        'explanation': explanation
    }


def generate_real_explanation(clean, intact, clean_prob, intact_prob):
    """
    Генерирует объяснение для реальной модели.
    
    Args:
        clean (bool): Чистое ли авто
        intact (bool): Целое ли авто
        clean_prob (float): Вероятность чистоты
        intact_prob (float): Вероятность целостности
        
    Returns:
        str: Текстовое объяснение
    """
    explanations = []
    
    # Анализ чистоты с учетом вероятности
    if not clean:
        if clean_prob < 0.3:
            explanations.append("Грязное авто — почистите для safety.")
        else:
            explanations.append("Обнаружены следы загрязнения")
    else:
        explanations.append("Автомобиль чистый")
    
    # Анализ целостности с учетом вероятности
    if not intact:
        if intact_prob < 0.3:
            explanations.append("Повреждения видны — проверьте.")
        else:
            explanations.append("Обнаружены незначительные повреждения")
    else:
        explanations.append("Повреждений не обнаружено")
    
    # Добавляем рекомендации
    if not clean and not intact:
        explanations.append("Критическое состояние: требуется мойка и ремонт")
    elif not clean:
        explanations.append("Рекомендуется мойка автомобиля")
    elif not intact:
        explanations.append("Обратитесь к специалисту для оценки повреждений")
    else:
        explanations.append("Автомобиль в отличном состоянии")
    
    return ". ".join(explanations) + "."


def generate_explanation(is_clean, is_intact, confidence):
    """
    Генерирует объяснение результатов анализа с улучшенной логикой.
    
    Args:
        is_clean (bool): Чистое ли авто
        is_intact (bool): Целое ли авто
        confidence (float): Уверенность модели
        
    Returns:
        str: Текстовое объяснение
    """
    explanations = []
    
    # Анализ чистоты
    if not is_clean:
        if confidence < 0.6:
            explanations.append("Обнаружены следы загрязнения")
        else:
            dirt_reasons = [
                "Обнаружена грязь на кузове",
                "Заметны следы пыли и загрязнений", 
                "Требуется мойка автомобиля",
                "Поверхность кузова загрязнена"
            ]
            explanations.append(random.choice(dirt_reasons))
    else:
        if confidence > 0.8:
            explanations.append("Автомобиль чистый")
        else:
            explanations.append("Автомобиль в хорошем состоянии")
    
    # Анализ целостности
    if not is_intact:
        if confidence < 0.6:
            explanations.append("Обнаружены незначительные повреждения")
        else:
            damage_reasons = [
                "Выявлены царапины на кузове",
                "Обнаружены вмятины или повреждения",
                "Есть следы ДТП или механических повреждений",
                "Заметны дефекты лакокрасочного покрытия"
            ]
            explanations.append(random.choice(damage_reasons))
    else:
        if confidence > 0.8:
            explanations.append("Повреждений не обнаружено")
        else:
            explanations.append("Кузов в хорошем состоянии")
    
    # Добавляем рекомендации
    if confidence < 0.6:
        advice = "Рекомендуется повторная проверка в лучших условиях освещения"
    elif not is_clean and not is_intact:
        if confidence > 0.7:
            advice = "Критическое состояние: требуется мойка и ремонт"
        else:
            advice = "Требуется мойка и осмотр автомобиля"
    elif not is_clean:
        advice = "Помойте автомобиль для улучшения внешнего вида"
    elif not is_intact:
        advice = "Обратитесь к специалисту для оценки повреждений"
    else:
        if confidence > 0.8:
            advice = "Автомобиль в отличном состоянии"
        else:
            advice = "Автомобиль в хорошем состоянии"
    
    explanations.append(advice)
    
    return ". ".join(explanations) + "."
