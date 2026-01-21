#!/usr/bin/env python3
"""
Language detection and localization utilities for StableAvatar.
Provides browser language detection and interface localization.
"""

import re
from typing import Dict, Any, Optional, List


def detect_browser_language(accept_language_header: str) -> str:
    """
    Detect the preferred language from browser Accept-Language header.
    
    Args:
        accept_language_header: The Accept-Language header from the browser
        
    Returns:
        str: Detected language code ('zh' for Chinese, 'en' for English, 'es' for Spanish, 'de' for German, 'ja' for Japanese, 'fr' for French, 'pt' for Portuguese, 'ru' for Russian, default 'zh')
    """
    if not accept_language_header:
        return "zh"  # Default to Chinese
    
    # Parse Accept-Language header (e.g., "zh-CN,zh;q=0.9,en;q=0.8")
    languages = []
    for lang_part in accept_language_header.split(','):
        lang_part = lang_part.strip()
        if ';' in lang_part:
            lang, quality = lang_part.split(';', 1)
            quality = float(quality.split('=')[1]) if 'q=' in quality else 1.0
        else:
            lang = lang_part
            quality = 1.0
        
        # Extract language code (e.g., "zh-CN" -> "zh", "en-US" -> "en")
        lang_code = lang.split('-')[0].lower()
        languages.append((lang_code, quality))
    
    # Sort by quality (higher first)
    languages.sort(key=lambda x: x[1], reverse=True)
    
    # Check for supported languages in order of preference
    for lang_code, _ in languages:
        if lang_code in ['zh', 'zh-cn', 'zh-tw', 'zh-hk']:
            return "zh"
        elif lang_code in ['en', 'en-us', 'en-gb', 'en-ca', 'en-au']:
            return "en"
        elif lang_code in ['es', 'es-es', 'es-mx', 'es-ar', 'es-co', 'es-pe', 'es-ve', 'es-cl', 'es-ec', 'es-gt', 'es-cu', 'es-bo', 'es-do', 'es-hn', 'es-py', 'es-sv', 'es-ni', 'es-cr', 'es-pa', 'es-pr', 'es-uy']:
            return "es"
        elif lang_code in ['de', 'de-de', 'de-at', 'de-ch', 'de-li', 'de-lu', 'de-be']:
            return "de"
        elif lang_code in ['ja', 'ja-jp']:
            return "ja"
        elif lang_code in ['fr', 'fr-fr', 'fr-ca', 'fr-be', 'fr-ch', 'fr-lu', 'fr-mc', 'fr-sn', 'fr-ci', 'fr-cm', 'fr-mg', 'fr-cd', 'fr-dj', 'fr-gn', 'fr-ml', 'fr-ne', 'fr-rw', 'fr-td', 'fr-tg', 'fr-bf', 'fr-bi', 'fr-km', 'fr-cf', 'fr-ga', 'fr-gq', 'fr-mr', 'fr-vu', 'fr-nc', 'fr-pf', 'fr-wf', 'fr-yt']:
            return "fr"
        elif lang_code in ['pt', 'pt-br', 'pt-pt', 'pt-ao', 'pt-mz', 'pt-cv', 'pt-gw', 'pt-st', 'pt-tl']:
            return "pt"
        elif lang_code in ['ru', 'ru-ru', 'ru-by', 'ru-kz', 'ru-kg', 'ru-tj', 'ru-tm', 'ru-uz', 'ru-md', 'ru-ua', 'ru-am', 'ru-az', 'ru-ge']:
            return "ru"
    
    # Default to Chinese if no match
    return "zh"


def get_language_from_request(request) -> str:
    """
    Extract language preference from a Gradio request object.
    
    Args:
        request: Gradio request object
        
    Returns:
        str: Detected language ('zh' or 'en')
    """
    try:
        # Try to get Accept-Language header
        accept_language = request.headers.get('Accept-Language', '')
        return detect_browser_language(accept_language)
    except:
        return "zh"  # Default fallback


def get_interface_texts(language: str) -> Dict[str, Dict[str, str]]:
    """
    Get interface texts for the specified language.
    
    Args:
        language: Language code ('zh', 'en', 'es', 'de', 'ja', 'fr', 'pt', or 'ru')
        
    Returns:
        Dict containing all interface texts
    """
    if language == "en":
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "Running on: {device_summary} | Device: {device} | Data Type: {dtype}",
                "language_label": "Language",
                "model_settings": "Model Settings",
                "video_generation": "Video Generation",
                "audio_extraction": "Audio Extraction", 
                "vocal_separation": "Vocal Separation"
            },
            "model_settings": {
                "gpu_memory_mode": "GPU Memory Mode",
                "gpu_memory_info": "Normal uses 25G VRAM, model_cpu_offload uses 13G VRAM",
                "teacache_threshold": "TeaCache Threshold",
                "teacache_info": "Recommended 0.1, 0 disables TeaCache acceleration",
                "num_skip_start_steps": "Skip Start Steps",
                "skip_steps_info": "Recommended 5",
                "clip_sample_n_frames": "Clip Sample Frames",
                "clip_frames_info": "Video frames, 81=2s@25fps, 161=4s@25fps, must be 4n+1",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)",
                "model_selection": "Transformer Model",
                "model_selection_info": "Choose the transformer model type: Square (standard) or Rec-Vec (recommended)"
            },
            "video_generation": {
                "upload_image": "Upload Image",
                "upload_audio": "Upload Audio",
                "prompt": "Prompt",
                "negative_prompt": "Negative Prompt",
                "negative_prompt_default": "vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, still image, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static image, cluttered background, three legs, many people in background, walking backwards",
                "start_generation": "🎬 Start Generation",
                "width": "Width",
                "height": "Height",
                "swap_dimensions": "🔄 Swap Width/Height",
                "adjust_size": "Adjust Size Based on Image",
                "guidance_scale": "Guidance Scale",
                "sampling_steps": "Sampling Steps (Recommended 50)",
                "text_guide_scale": "Text Guidance Scale",
                "audio_guide_scale": "Audio Guidance Scale",
                "motion_frame": "Motion Frame",
                "fps": "FPS",
                "overlap_window_length": "Overlap Window Length",
                "seed": "Seed (positive integer, -1 for random)",
                "status": "Status",
                "generated_result": "Generated Result",
                "seed_output": "Seed"
            },
            "audio_extraction": {
                "upload_video": "Upload Video",
                "start_extraction": "🎬 Start Extraction",
                "status": "Status",
                "generated_result": "Generated Result"
            },
            "vocal_separation": {
                "upload_audio": "Upload Audio",
                "start_separation": "🎬 Start Separation",
                "status": "Status",
                "generated_result": "Generated Result"
            }
        }
    elif language == "es":  # Spanish
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "Ejecutando en: {device_summary} | Dispositivo: {device} | Tipo de datos: {dtype}",
                "language_label": "Idioma",
                "model_settings": "Configuración del Modelo",
                "video_generation": "Generación de Video",
                "audio_extraction": "Extracción de Audio", 
                "vocal_separation": "Separación Vocal"
            },
            "model_settings": {
                "gpu_memory_mode": "Modo de Memoria GPU",
                "gpu_memory_info": "Normal usa 25G VRAM, model_cpu_offload usa 13G VRAM",
                "teacache_threshold": "Umbral TeaCache",
                "teacache_info": "Recomendado 0.1, 0 desactiva la aceleración TeaCache",
                "num_skip_start_steps": "Omitir Pasos Iniciales",
                "skip_steps_info": "Recomendado 5",
                "clip_sample_n_frames": "Frames de Muestra Clip",
                "clip_frames_info": "Frames de video, 81=2s@25fps, 161=4s@25fps, debe ser 4n+1",
                "model_selection": "Modelo Transformer",
                "model_selection_info": "Elige el tipo de modelo transformer: Square (estándar) o Rec-Vec (recomendado)"
            },
            "video_generation": {
                "upload_image": "Subir Imagen",
                "upload_audio": "Subir Audio",
                "prompt": "Prompt",
                "negative_prompt": "Prompt Negativo",
                "negative_prompt_default": "colores vivos, sobreexpuesto, estático, detalles borrosos, subtítulos, estilo, obra de arte, pintura, imagen fija, gris general, peor calidad, baja calidad, artefactos de compresión JPEG, feo, incompleto, dedos extra, manos mal dibujadas, cara mal dibujada, deforme, desfigurado, extremidades malformadas, dedos fusionados, imagen estática, fondo desordenado, tres piernas, mucha gente en el fondo, caminando hacia atrás",
                "start_generation": "🎬 Iniciar Generación",
                "width": "Ancho",
                "height": "Alto",
                "swap_dimensions": "🔄 Intercambiar Ancho/Alto",
                "adjust_size": "Ajustar Tamaño Basado en Imagen",
                "guidance_scale": "Escala de Guía",
                "sampling_steps": "Pasos de Muestreo (Recomendado 50)",
                "text_guide_scale": "Escala de Guía de Texto",
                "audio_guide_scale": "Escala de Guía de Audio",
                "motion_frame": "Frame de Movimiento",
                "fps": "FPS",
                "overlap_window_length": "Longitud de Ventana de Solapamiento",
                "seed": "Semilla (entero positivo, -1 para aleatorio)",
                "status": "Estado",
                "generated_result": "Resultado Generado",
                "seed_output": "Semilla"
            },
            "audio_extraction": {
                "upload_video": "Subir Video",
                "start_extraction": "🎬 Iniciar Extracción",
                "status": "Estado",
                "generated_result": "Resultado Generado"
            },
            "vocal_separation": {
                "upload_audio": "Subir Audio",
                "start_separation": "🎬 Iniciar Separación",
                "status": "Estado",
                "generated_result": "Resultado Generado"
            }
        }
    elif language == "de":  # German
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "Läuft auf: {device_summary} | Gerät: {device} | Datentyp: {dtype}",
                "language_label": "Sprache",
                "model_settings": "Modelleinstellungen",
                "video_generation": "Video-Generierung",
                "audio_extraction": "Audio-Extraktion", 
                "vocal_separation": "Gesangstrennung"
            },
            "model_settings": {
                "gpu_memory_mode": "GPU-Speichermodus",
                "gpu_memory_info": "Normal verwendet 25G VRAM, model_cpu_offload verwendet 13G VRAM",
                "teacache_threshold": "TeaCache-Schwellenwert",
                "teacache_info": "Empfohlen 0.1, 0 deaktiviert TeaCache-Beschleunigung",
                "num_skip_start_steps": "Startschritte Überspringen",
                "skip_steps_info": "Empfohlen 5",
                "clip_sample_n_frames": "Clip-Sample-Frames",
                "clip_frames_info": "Video-Frames, 81=2s@25fps, 161=4s@25fps, muss 4n+1 sein",
                "model_selection": "Transformer-Modell",
                "model_selection_info": "Wählen Sie den Transformer-Modelltyp: Square (Standard) oder Rec-Vec (empfohlen)"
            },
            "video_generation": {
                "upload_image": "Bild Hochladen",
                "upload_audio": "Audio Hochladen",
                "prompt": "Prompt",
                "negative_prompt": "Negativer Prompt",
                "negative_prompt_default": "lebendige Farben, überbelichtet, statisch, unscharfe Details, Untertitel, Stil, Kunstwerk, Gemälde, Standbild, insgesamt grau, schlechteste Qualität, niedrige Qualität, JPEG-Komprimierungsartefakte, hässlich, unvollständig, zusätzliche Finger, schlecht gezeichnete Hände, schlecht gezeichnetes Gesicht, deformiert, entstellt, missgebildete Gliedmaßen, verschmolzene Finger, statisches Bild, unordentlicher Hintergrund, drei Beine, viele Menschen im Hintergrund, rückwärts gehend",
                "start_generation": "🎬 Generierung Starten",
                "width": "Breite",
                "height": "Höhe",
                "swap_dimensions": "🔄 Breite/Höhe Tauschen",
                "adjust_size": "Größe Basierend auf Bild Anpassen",
                "guidance_scale": "Führungsskala",
                "sampling_steps": "Sampling-Schritte (Empfohlen 50)",
                "text_guide_scale": "Text-Führungsskala",
                "audio_guide_scale": "Audio-Führungsskala",
                "motion_frame": "Bewegungsframe",
                "fps": "FPS",
                "overlap_window_length": "Überlappungsfenster-Länge",
                "seed": "Seed (positive Ganzzahl, -1 für zufällig)",
                "status": "Status",
                "generated_result": "Generiertes Ergebnis",
                "seed_output": "Seed"
            },
            "audio_extraction": {
                "upload_video": "Video Hochladen",
                "start_extraction": "🎬 Extraktion Starten",
                "status": "Status",
                "generated_result": "Generiertes Ergebnis"
            },
            "vocal_separation": {
                "upload_audio": "Audio Hochladen",
                "start_separation": "🎬 Trennung Starten",
                "status": "Status",
                "generated_result": "Generiertes Ergebnis"
            }
        }
    elif language == "ja":  # Japanese
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "実行環境: {device_summary} | デバイス: {device} | データ型: {dtype}",
                "language_label": "言語",
                "model_settings": "モデル設定",
                "video_generation": "動画生成",
                "audio_extraction": "音声抽出", 
                "vocal_separation": "ボーカル分離"
            },
            "model_settings": {
                "gpu_memory_mode": "GPUメモリモード",
                "gpu_memory_info": "Normalは25G VRAM、model_cpu_offloadは13G VRAMを使用",
                "teacache_threshold": "TeaCache閾値",
                "teacache_info": "推奨値0.1、0でTeaCache加速を無効化",
                "num_skip_start_steps": "開始ステップをスキップ",
                "skip_steps_info": "推奨値5",
                "clip_sample_n_frames": "Clipサンプルフレーム",
                "clip_frames_info": "動画フレーム、81=2秒@25fps、161=4秒@25fps、4n+1である必要があります",
                "model_selection": "Transformerモデル",
                "model_selection_info": "Transformerモデルタイプを選択: Square（標準）またはRec-Vec（推奨）"
            },
            "video_generation": {
                "upload_image": "画像をアップロード",
                "upload_audio": "音声をアップロード",
                "prompt": "プロンプト",
                "negative_prompt": "ネガティブプロンプト",
                "negative_prompt_default": "鮮やかな色、露出オーバー、静止、ぼやけた詳細、字幕、スタイル、アートワーク、絵画、静止画像、全体的にグレー、最悪の品質、低品質、JPEG圧縮アーティファクト、醜い、不完全、余分な指、不適切に描かれた手、不適切に描かれた顔、変形、破損、奇形の手足、融合した指、静止画像、乱雑な背景、3本足、背景に多くの人、後ろ向きに歩く",
                "start_generation": "🎬 生成開始",
                "width": "幅",
                "height": "高さ",
                "swap_dimensions": "🔄 幅/高さを交換",
                "adjust_size": "画像に基づいてサイズを調整",
                "guidance_scale": "ガイダンススケール",
                "sampling_steps": "サンプリングステップ（推奨50）",
                "text_guide_scale": "テキストガイダンススケール",
                "audio_guide_scale": "音声ガイダンススケール",
                "motion_frame": "モーションフレーム",
                "fps": "FPS",
                "overlap_window_length": "オーバーラップウィンドウ長",
                "seed": "シード（正の整数、-1でランダム）",
                "status": "ステータス",
                "generated_result": "生成結果",
                "seed_output": "シード"
            },
            "audio_extraction": {
                "upload_video": "動画をアップロード",
                "start_extraction": "🎬 抽出開始",
                "status": "ステータス",
                "generated_result": "生成結果"
            },
            "vocal_separation": {
                "upload_audio": "音声をアップロード",
                "start_separation": "🎬 分離開始",
                "status": "ステータス",
                "generated_result": "生成結果"
            }
        }
    elif language == "fr":  # French
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "Exécution sur: {device_summary} | Appareil: {device} | Type de données: {dtype}",
                "language_label": "Langue",
                "model_settings": "Paramètres du Modèle",
                "video_generation": "Génération Vidéo",
                "audio_extraction": "Extraction Audio", 
                "vocal_separation": "Séparation Vocale"
            },
            "model_settings": {
                "gpu_memory_mode": "Mode Mémoire GPU",
                "gpu_memory_info": "Normal utilise 25G VRAM, model_cpu_offload utilise 13G VRAM",
                "teacache_threshold": "Seuil TeaCache",
                "teacache_info": "Recommandé 0.1, 0 désactive l'accélération TeaCache",
                "num_skip_start_steps": "Ignorer les Étapes Initiales",
                "skip_steps_info": "Recommandé 5",
                "clip_sample_n_frames": "Images d'Échantillon Clip",
                "clip_frames_info": "Images vidéo, 81=2s@25fps, 161=4s@25fps, doit être 4n+1",
                "model_selection": "Modèle Transformer",
                "model_selection_info": "Choisissez le type de modèle transformer: Square (standard) ou Rec-Vec (recommandé)"
            },
            "video_generation": {
                "upload_image": "Télécharger Image",
                "upload_audio": "Télécharger Audio",
                "prompt": "Prompt",
                "negative_prompt": "Prompt Négatif",
                "negative_prompt_default": "couleurs vives, surexposé, statique, détails flous, sous-titres, style, œuvre d'art, peinture, image fixe, gris général, pire qualité, basse qualité, artefacts de compression JPEG, laid, incomplet, doigts supplémentaires, mains mal dessinées, visage mal dessiné, déformé, défiguré, membres malformés, doigts fusionnés, image statique, arrière-plan encombré, trois jambes, beaucoup de gens en arrière-plan, marchant à reculons",
                "start_generation": "🎬 Démarrer Génération",
                "width": "Largeur",
                "height": "Hauteur",
                "swap_dimensions": "🔄 Échanger Largeur/Hauteur",
                "adjust_size": "Ajuster Taille Basée sur Image",
                "guidance_scale": "Échelle de Guidage",
                "sampling_steps": "Étapes d'Échantillonnage (Recommandé 50)",
                "text_guide_scale": "Échelle de Guidage Texte",
                "audio_guide_scale": "Échelle de Guidage Audio",
                "motion_frame": "Image de Mouvement",
                "fps": "FPS",
                "overlap_window_length": "Longueur Fenêtre de Chevauchement",
                "seed": "Graine (entier positif, -1 pour aléatoire)",
                "status": "Statut",
                "generated_result": "Résultat Généré",
                "seed_output": "Graine"
            },
            "audio_extraction": {
                "upload_video": "Télécharger Vidéo",
                "start_extraction": "🎬 Démarrer Extraction",
                "status": "Statut",
                "generated_result": "Résultat Généré"
            },
            "vocal_separation": {
                "upload_audio": "Télécharger Audio",
                "start_separation": "🎬 Démarrer Séparation",
                "status": "Statut",
                "generated_result": "Résultat Généré"
            }
        }
    elif language == "pt":  # Portuguese
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "Executando em: {device_summary} | Dispositivo: {device} | Tipo de dados: {dtype}",
                "language_label": "Idioma",
                "model_settings": "Configurações do Modelo",
                "video_generation": "Geração de Vídeo",
                "audio_extraction": "Extração de Áudio", 
                "vocal_separation": "Separação Vocal"
            },
            "model_settings": {
                "gpu_memory_mode": "Modo de Memória GPU",
                "gpu_memory_info": "Normal usa 25G VRAM, model_cpu_offload usa 13G VRAM",
                "teacache_threshold": "Limite TeaCache",
                "teacache_info": "Recomendado 0.1, 0 desativa aceleração TeaCache",
                "num_skip_start_steps": "Pular Passos Iniciais",
                "skip_steps_info": "Recomendado 5",
                "clip_sample_n_frames": "Quadros de Amostra Clip",
                "clip_frames_info": "Quadros de vídeo, 81=2s@25fps, 161=4s@25fps, deve ser 4n+1",
                "model_selection": "Modelo Transformer",
                "model_selection_info": "Escolha o tipo de modelo transformer: Square (padrão) ou Rec-Vec (recomendado)"
            },
            "video_generation": {
                "upload_image": "Carregar Imagem",
                "upload_audio": "Carregar Áudio",
                "prompt": "Prompt",
                "negative_prompt": "Prompt Negativo",
                "negative_prompt_default": "cores vivas, superexposto, estático, detalhes borrados, legendas, estilo, obra de arte, pintura, imagem fixa, cinza geral, pior qualidade, baixa qualidade, artefatos de compressão JPEG, feio, incompleto, dedos extras, mãos mal desenhadas, rosto mal desenhado, deformado, desfigurado, membros malformados, dedos fundidos, imagem estática, fundo desordenado, três pernas, muitas pessoas no fundo, andando para trás",
                "start_generation": "🎬 Iniciar Geração",
                "width": "Largura",
                "height": "Altura",
                "swap_dimensions": "🔄 Trocar Largura/Altura",
                "adjust_size": "Ajustar Tamanho Baseado na Imagem",
                "guidance_scale": "Escala de Orientação",
                "sampling_steps": "Passos de Amostragem (Recomendado 50)",
                "text_guide_scale": "Escala de Orientação de Texto",
                "audio_guide_scale": "Escala de Orientação de Áudio",
                "motion_frame": "Quadro de Movimento",
                "fps": "FPS",
                "overlap_window_length": "Comprimento da Janela de Sobreposição",
                "seed": "Semente (inteiro positivo, -1 para aleatório)",
                "status": "Status",
                "generated_result": "Resultado Gerado",
                "seed_output": "Semente"
            },
            "audio_extraction": {
                "upload_video": "Carregar Vídeo",
                "start_extraction": "🎬 Iniciar Extração",
                "status": "Status",
                "generated_result": "Resultado Gerado"
            },
            "vocal_separation": {
                "upload_audio": "Carregar Áudio",
                "start_separation": "🎬 Iniciar Separação",
                "status": "Status",
                "generated_result": "Resultado Gerado"
            }
        }
    elif language == "ru":  # Russian
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "Запуск на: {device_summary} | Устройство: {device} | Тип данных: {dtype}",
                "language_label": "Язык",
                "model_settings": "Настройки Модели",
                "video_generation": "Генерация Видео",
                "audio_extraction": "Извлечение Аудио", 
                "vocal_separation": "Разделение Вокала"
            },
            "model_settings": {
                "gpu_memory_mode": "Режим Памяти GPU",
                "gpu_memory_info": "Normal использует 25G VRAM, model_cpu_offload использует 13G VRAM",
                "teacache_threshold": "Порог TeaCache",
                "teacache_info": "Рекомендуется 0.1, 0 отключает ускорение TeaCache",
                "num_skip_start_steps": "Пропустить Начальные Шаги",
                "skip_steps_info": "Рекомендуется 5",
                "clip_sample_n_frames": "Кадры Образца Clip",
                "clip_frames_info": "Видеокадры, 81=2с@25fps, 161=4с@25fps, должно быть 4n+1",
                "model_selection": "Модель Transformer",
                "model_selection_info": "Выберите тип модели transformer: Square (стандартная) или Rec-Vec (рекомендуемая)"
            },
            "video_generation": {
                "upload_image": "Загрузить Изображение",
                "upload_audio": "Загрузить Аудио",
                "prompt": "Промпт",
                "negative_prompt": "Негативный Промпт",
                "negative_prompt_default": "яркие цвета, переэкспонированный, статичный, размытые детали, субтитры, стиль, произведение искусства, живопись, неподвижное изображение, общий серый, худшее качество, низкое качество, артефакты сжатия JPEG, уродливый, неполный, лишние пальцы, плохо нарисованные руки, плохо нарисованное лицо, деформированный, обезображенный, неправильно сформированные конечности, сросшиеся пальцы, статичное изображение, загроможденный фон, три ноги, много людей на фоне, идущий назад",
                "start_generation": "🎬 Начать Генерацию",
                "width": "Ширина",
                "height": "Высота",
                "swap_dimensions": "🔄 Поменять Ширину/Высоту",
                "adjust_size": "Настроить Размер на Основе Изображения",
                "guidance_scale": "Шкала Направления",
                "sampling_steps": "Шаги Семплирования (Рекомендуется 50)",
                "text_guide_scale": "Шкала Направления Текста",
                "audio_guide_scale": "Шкала Направления Аудио",
                "motion_frame": "Кадр Движения",
                "fps": "FPS",
                "overlap_window_length": "Длина Окна Перекрытия",
                "seed": "Семя (положительное целое, -1 для случайного)",
                "status": "Статус",
                "generated_result": "Сгенерированный Результат",
                "seed_output": "Семя"
            },
            "audio_extraction": {
                "upload_video": "Загрузить Видео",
                "start_extraction": "🎬 Начать Извлечение",
                "status": "Статус",
                "generated_result": "Сгенерированный Результат"
            },
            "vocal_separation": {
                "upload_audio": "Загрузить Аудио",
                "start_separation": "🎬 Начать Разделение",
                "status": "Статус",
                "generated_result": "Сгенерированный Результат"
            }
        }
    else:  # Chinese (zh)
        return {
            "main": {
                "title": "StableAvatar",
                "device_info": "运行环境: {device_summary} | 设备: {device} | 数据类型: {dtype}",
                "language_label": "语言",
                "model_settings": "模型设置",
                "video_generation": "视频生成",
                "audio_extraction": "音频提取",
                "vocal_separation": "人声分离"
            },
            "model_settings": {
                "gpu_memory_mode": "显存模式",
                "gpu_memory_info": "Normal占用25G显存，model_cpu_offload占用13G显存",
                "teacache_threshold": "teacache threshold",
                "teacache_info": "推荐参数0.1，0为禁用teacache加速",
                "num_skip_start_steps": "跳过开始步数",
                "skip_steps_info": "推荐参数5",
                "clip_sample_n_frames": "Clip采样帧数",
                "clip_frames_info": "视频帧数，81=2秒@25fps，161=4秒@25fps，必须为4n+1",
                "model_selection": "Transformer模型",
                "model_selection_info": "选择transformer模型类型：Square（标准）或Rec-Vec（推荐）"
            },
            "video_generation": {
                "upload_image": "上传图片",
                "upload_audio": "上传音频",
                "prompt": "提示词",
                "negative_prompt": "负面提示词",
                "negative_prompt_default": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "start_generation": "🎬 开始生成",
                "width": "宽度",
                "height": "高度",
                "swap_dimensions": "🔄 交换宽高",
                "adjust_size": "根据图片调整宽高",
                "guidance_scale": "guidance scale",
                "sampling_steps": "采样步数（推荐50步）",
                "text_guide_scale": "text guidance scale",
                "audio_guide_scale": "audio guidance scale",
                "motion_frame": "motion frame",
                "fps": "帧率",
                "overlap_window_length": "overlap window length",
                "seed": "种子，请输入正整数，-1为随机",
                "status": "提示信息",
                "generated_result": "生成结果",
                "seed_output": "种子"
            },
            "audio_extraction": {
                "upload_video": "上传视频",
                "start_extraction": "🎬 开始提取",
                "status": "提示信息",
                "generated_result": "生成结果"
            },
            "vocal_separation": {
                "upload_audio": "上传音频",
                "start_separation": "🎬 开始分离",
                "status": "提示信息",
                "generated_result": "生成结果"
            }
        }


def get_display_language(language_code: str) -> str:
    """
    Convert language code to display name.
    
    Args:
        language_code: Language code ('zh', 'en', 'es', 'de', 'ja', 'fr', 'pt', or 'ru')
        
    Returns:
        str: Display name ('中文', 'English', 'Español', 'Deutsch', '日本語', 'Français', 'Português', or 'Русский')
    """
    language_map = {
        "zh": "中文",
        "en": "English", 
        "es": "Español",
        "de": "Deutsch",
        "ja": "日本語",
        "fr": "Français",
        "pt": "Português",
        "ru": "Русский"
    }
    return language_map.get(language_code, "中文")


def get_language_choices() -> List[tuple]:
    """
    Get language choices for Gradio Radio component.
    
    Returns:
        List of tuples (display_name, language_code)
    """
    return [
        ("中文", "zh"), 
        ("English", "en"), 
        ("Español", "es"), 
        ("Deutsch", "de"),
        ("日本語", "ja"),
        ("Français", "fr"),
        ("Português", "pt"),
        ("Русский", "ru")
    ]


def create_language_detection_js() -> str:
    """
    Create simple JavaScript code for client-side language detection.
    This version only sets the radio button without triggering events to avoid conflicts.
    
    Returns:
        str: JavaScript code for language detection
    """
    return """
    <script>
    function detectLanguage() {
        const language = navigator.language || navigator.userLanguage;
        const lang = language.toLowerCase();
        
        let langCode = 'en'; // default to English
        if (lang.startsWith('zh')) {
            langCode = 'zh';
        } else if (lang.startsWith('en')) {
            langCode = 'en';
        } else if (lang.startsWith('es')) {
            langCode = 'es';
        } else if (lang.startsWith('de')) {
            langCode = 'de';
        } else if (lang.startsWith('ja')) {
            langCode = 'ja';
        } else if (lang.startsWith('fr')) {
            langCode = 'fr';
        } else if (lang.startsWith('pt')) {
            langCode = 'pt';
        } else if (lang.startsWith('ru')) {
            langCode = 'ru';
        }
        
        // Map language codes to display names
        const langMap = {
            'zh': '中文',
            'en': 'English',
            'es': 'Español',
            'de': 'Deutsch',
            'ja': '日本語',
            'fr': 'Français',
            'pt': 'Português',
            'ru': 'Русский'
        };
        
        const displayName = langMap[langCode];
        console.log('Detected language:', language, '->', langCode, '->', displayName);
        
        // Simple function to set language without triggering events
        function setLanguage() {
            const radioButtons = document.querySelectorAll('input[type="radio"]');
            radioButtons.forEach(radio => {
                if (radio.value === displayName) {
                    console.log('Setting language to:', displayName);
                    radio.checked = true;
                    // Don't trigger any events to avoid conflicts
                }
            });
        }
        
        // Try to set language after a delay
        setTimeout(setLanguage, 2000);
    }
    
    // Run language detection when the page loads
    document.addEventListener('DOMContentLoaded', detectLanguage);
    </script>
    """


if __name__ == "__main__":
    # Test the language detection
    test_headers = [
        "zh-CN,zh;q=0.9,en;q=0.8",
        "en-US,en;q=0.9",
        "es-ES,es;q=0.9,en;q=0.8",
        "de-DE,de;q=0.9,en;q=0.8",
        "ja-JP,ja;q=0.9,en;q=0.8",
        "fr-FR,fr;q=0.9,en;q=0.8",
        "pt-BR,pt;q=0.9,en;q=0.8",
        "ru-RU,ru;q=0.9,en;q=0.8",
        "zh-TW,zh;q=0.9"
    ]
    
    print("Testing language detection:")
    for header in test_headers:
        detected = detect_browser_language(header)
        display_name = get_display_language(detected)
        print(f"Header: {header} -> Detected: {detected} ({display_name})")
    
    print("\nTesting interface texts:")
    for lang in ["zh", "en", "es", "de", "ja", "fr", "pt", "ru"]:
        texts = get_interface_texts(lang)
        display_name = get_display_language(lang)
        print(f"\n{display_name} ({lang.upper()}):")
        print(f"  Title: {texts['main']['title']}")
        print(f"  Language Label: {texts['main']['language_label']}")
        print(f"  Start Generation: {texts['video_generation']['start_generation']}")
        print(f"  Upload Image: {texts['video_generation']['upload_image']}")
    
    print("\nTesting language choices:")
    choices = get_language_choices()
    for display_name, lang_code in choices:
        print(f"  {display_name} -> {lang_code}")
