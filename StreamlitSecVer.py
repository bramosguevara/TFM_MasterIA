import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import re
import hashlib
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

st.set_page_config(
    page_title="Generador de Cartas Clash Royale IA",
    page_icon="🎮",
    layout="wide"
)

def extract_exact_numbers_from_prompt(prompt):
    """Extrae EXACTAMENTE los números que especifica el usuario"""
    prompt_lower = prompt.lower()
    extracted = {'cost': None, 'damage': None, 'health': None, 'duration': None}
    
    # Patrones para coste
    cost_patterns = [
        r'(?:cueste?|coste?|cost[eo]?|elixir)\s*(?:de)?\s*(\d+)',
        r'(\d+)\s*(?:de\s*)?(?:elixir|coste?|cost[eo]?)',
        r'(?:que\s*)?(?:cueste?|valga)\s*(\d+)',
        r'de\s*(\d+)\s*elixir'
    ]
    
    # Patrones para daño
    damage_patterns = [
        r'(\d+)\s*(?:de\s*)?(?:daño|damage|ataque|attack)',
        r'(?:daño|damage|ataque|attack)\s*(?:de)?\s*(\d+)',
        r'(?:que\s*)?(?:haga|cause|tenga)\s*(\d+)\s*(?:de\s*)?(?:daño|damage|ataque)',
        r'con\s*(\d+)\s*(?:de\s*)?(?:daño|ataque)'
    ]
    
    # Patrones para vida
    health_patterns = [
        r'(\d+)\s*(?:de\s*)?(?:vida|health|hp|salud|resistencia)',
        r'(?:vida|health|hp|salud|resistencia)\s*(?:de)?\s*(\d+)',
        r'(?:que\s*)?(?:tenga|posea)\s*(\d+)\s*(?:de\s*)?(?:vida|health|hp)',
        r'y\s*(\d+)\s*(?:de\s*)?(?:vida|health)'
    ]
    
    # Patrones para duración
    duration_patterns = [
        r'duracion\s*(?:de)?\s*(\d+)\s*(?:segundos?|segs?|s)',
        r'dure\s*(\d+)\s*(?:segundos?|segs?|s)',
        r'(\d+)\s*segundos?\s*(?:de\s*)?(?:duracion|efecto|tiempo)',
        r'(?:por|durante)\s*(\d+)\s*(?:segundos?|segs?|s)',
        r'efecto\s*(?:de)?\s*(\d+)\s*(?:segundos?|segs?|s)',
        r'con\s*duracion\s*(?:de)?\s*(\d+)\s*(?:segundos?|segs?|s)',
        r'que\s*dure\s*(\d+)\s*(?:segundos?|segs?|s)'
    ]
    
    # Extraer números exactos
    for pattern in cost_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            extracted['cost'] = int(match.group(1))
            break
    
    for pattern in damage_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            extracted['damage'] = int(match.group(1))
            break
    
    for pattern in health_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            extracted['health'] = int(match.group(1))
            break
    
    for pattern in duration_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            extracted['duration'] = int(match.group(1))
            break
    
    return extracted

def detect_character_and_type(prompt):
    """Detecta el personaje específico y tipo de carta"""
    prompt_lower = prompt.lower()
    
    # Personajes específicos con info completa
    characters = {
        'golem': {'type': 'Troops and Defenses', 'base_cost': 8, 'character': 'Golem', 'base_duration': 0},
        'gigante': {'type': 'Troops and Defenses', 'base_cost': 5, 'character': 'Gigante', 'base_duration': 0},
        'caballero': {'type': 'Troops and Defenses', 'base_cost': 3, 'character': 'Caballero', 'base_duration': 0},
        'arquero': {'type': 'Troops and Defenses', 'base_cost': 3, 'character': 'Arquero', 'base_duration': 0},
        'dragon': {'type': 'Troops and Defenses', 'base_cost': 4, 'character': 'Dragón', 'base_duration': 0},
        'mago': {'type': 'Troops and Defenses', 'base_cost': 5, 'character': 'Mago', 'base_duration': 0},
        
        # Hechizos específicos
        'hechizo': {'type': 'Damaging Spells', 'base_cost': 4, 'character': 'Hechizo', 'base_duration': 3},
        'rayo': {'type': 'Damaging Spells', 'base_cost': 6, 'character': 'Rayo', 'base_duration': 1},
        'bola de fuego': {'type': 'Damaging Spells', 'base_cost': 4, 'character': 'Bola de Fuego', 'base_duration': 2},
        'flecha': {'type': 'Damaging Spells', 'base_cost': 3, 'character': 'Flechas', 'base_duration': 1},
        'veneno': {'type': 'Damaging Spells', 'base_cost': 4, 'character': 'Veneno', 'base_duration': 8},
        'ralentizar': {'type': 'Damaging Spells', 'base_cost': 2, 'character': 'Ralentizar', 'base_duration': 5},
        'congelar': {'type': 'Damaging Spells', 'base_cost': 4, 'character': 'Congelar', 'base_duration': 4},
        
        # Edificios
        'torre': {'type': 'Spawners', 'base_cost': 4, 'character': 'Torre', 'base_duration': 0},
        'cañon': {'type': 'Spawners', 'base_cost': 3, 'character': 'Cañón', 'base_duration': 0},
        'mortero': {'type': 'Spawners', 'base_cost': 4, 'character': 'Mortero', 'base_duration': 0},
    }
    
    for keyword, info in characters.items():
        if keyword in prompt_lower:
            return info
    
    # Detección por tipo general
    if any(word in prompt_lower for word in ['hechizo', 'spell', 'magia', 'conjuro']):
        return {'type': 'Damaging Spells', 'base_cost': 4, 'character': 'Hechizo', 'base_duration': 3}
    elif any(word in prompt_lower for word in ['edificio', 'torre', 'defensa', 'spawner']):
        return {'type': 'Spawners', 'base_cost': 4, 'character': 'Torre', 'base_duration': 0}
    else:
        return {'type': 'Troops and Defenses', 'base_cost': 4, 'character': 'Guerrero', 'base_duration': 0}

def detect_elements(prompt):
    """Detecta elementos mágicos específicos"""
    prompt_lower = prompt.lower()
    elements = []
    
    if any(word in prompt_lower for word in ['hielo', 'nieve', 'congelar', 'frio']):
        elements.append('hielo')
    if any(word in prompt_lower for word in ['fuego', 'llama', 'quemar', 'ardiente']):
        elements.append('fuego')
    if any(word in prompt_lower for word in ['rayo', 'electrico', 'trueno']):
        elements.append('electrico')
    if any(word in prompt_lower for word in ['veneno', 'toxico']):
        elements.append('veneno')
    
    return elements

def parse_user_prompt_precisely(prompt):
    """Parser que RESPETA EXACTAMENTE el prompt del usuario"""
    # Extraer números específicos del usuario
    numbers = extract_exact_numbers_from_prompt(prompt)
    
    # Detectar personaje y tipo
    char_info = detect_character_and_type(prompt)
    
    # Detectar elementos mágicos
    elements = detect_elements(prompt)
    
    # USAR EXACTAMENTE los números especificados por el usuario
    cost = numbers['cost']
    damage = numbers['damage']
    health = numbers['health']
    duration = numbers['duration']
    
    # Solo usar valores por defecto si NO fueron especificados por el usuario
    if cost is None:
        cost = char_info['base_cost']
    
    if damage is None and char_info['type'] != 'Spawners':
        if char_info['type'] == 'Damaging Spells':
            damage = cost * 70
        else:
            damage = cost * 50
    elif damage is None:
        damage = 0
    
    if health is None and char_info['type'] != 'Damaging Spells':
        if char_info['type'] == 'Spawners':
            health = cost * 300
        else:
            health = cost * 150
    elif health is None:
        health = 0
    
    # Duración solo para hechizos
    if char_info['type'] == 'Damaging Spells':
        if duration is None:
            duration = char_info.get('base_duration', 3)
    else:
        duration = 0
    
    return {
        'Cost': cost,
        'Damage': damage,
        'Health (+Shield)': health,
        'Duration': duration,
        'Type': char_info['type'],
        'Character': char_info['character'],
        'Elements': elements,
        'original_prompt': prompt
    }

def generate_precise_narrative(card_data):
    """Genera narrativa que coincide EXACTAMENTE con el prompt"""
    original_prompt = card_data.get('original_prompt', '').lower()
    cost = int(card_data.get('Cost', 3))
    damage = int(card_data.get('Damage', 0))
    health = int(card_data.get('Health (+Shield)', 0))
    duration = int(card_data.get('Duration', 0))
    character = card_data.get('Character', 'Guerrero')
    elements = card_data.get('Elements', [])
    card_type = card_data.get('Type', 'Troops and Defenses')
    
    # Construir narrativa basada en el prompt ESPECÍFICO
    narrative_parts = []
    
    # Introducción con personaje específico
    if 'hielo' in elements:
        narrative_parts.append(f"¡{character} de hielo de {cost} elixir!")
    elif 'fuego' in elements:
        narrative_parts.append(f"¡{character} de fuego de {cost} elixir!")
    elif 'electrico' in elements:
        narrative_parts.append(f"¡{character} eléctrico de {cost} elixir!")
    else:
        narrative_parts.append(f"¡{character} de {cost} elixir!")
    
    # Descripción de elementos
    if 'hielo' in elements:
        narrative_parts.append("Congela a sus enemigos con poder glacial.")
    elif 'fuego' in elements:
        narrative_parts.append("Arde con llamas devastadoras que consumen todo.")
    elif 'electrico' in elements:
        narrative_parts.append("Electriza el campo con descargas letales.")
    
    # Estadísticas REALES del usuario
    if card_type == 'Damaging Spells':
        narrative_parts.append(f"Causa {damage} puntos de daño")
        if duration > 0:
            if duration == 1:
                narrative_parts.append("con efecto instantáneo.")
            elif duration <= 3:
                narrative_parts.append(f"con efecto que dura {duration} segundos.")
            else:
                narrative_parts.append(f"manteniendo su efecto por {duration} segundos.")
        else:
            narrative_parts.append("devastador y preciso.")
    elif card_type == 'Spawners':
        narrative_parts.append(f"Con {health} puntos de resistencia")
        if health >= 1000:
            narrative_parts.append("es prácticamente indestructible.")
        else:
            narrative_parts.append("defiende eficazmente.")
    else:  # Troops
        narrative_parts.append(f"Ataque de {damage} y resistencia de {health}")
        if damage > 300 and health > 600:
            narrative_parts.append("- ¡Una bestia imparable!")
        elif damage > 300:
            narrative_parts.append("con golpes devastadores.")
        elif health > 800:
            narrative_parts.append("como un tanque blindado.")
        else:
            narrative_parts.append("perfectamente balanceado.")
    
    # Final apropiado
    if cost >= 7:
        narrative_parts.append("¡Dominará completamente la arena!")
    elif cost >= 4:
        narrative_parts.append("¡Perfecto para estrategias épicas!")
    else:
        narrative_parts.append("¡Ideal para ciclos rápidos!")
    
    return " ".join(narrative_parts)

def generate_precise_card_name(card_data):
    """Genera nombre basado en el prompt específico"""
    character = card_data.get('Character', 'Guerrero')
    elements = card_data.get('Elements', [])
    cost = card_data.get('Cost', 3)
    
    name_parts = [character.upper()]
    
    if 'hielo' in elements:
        name_parts.append('DE HIELO')
    elif 'fuego' in elements:
        name_parts.append('DE FUEGO')
    elif 'electrico' in elements:
        name_parts.append('ELÉCTRICO')
    
    # Sufijo por rareza
    if cost >= 7:
        name_parts.append('SUPREMO')
    elif cost >= 5:
        name_parts.append('ÉPICO')
    elif cost >= 3:
        name_parts.append('ELITE')
    else:
        name_parts.append('RÁPIDO')
    
    return ' '.join(name_parts)

class StableDiffusionCardGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    @st.cache_resource(show_spinner="Cargando modelo de difusión...")
    def setup_stable_diffusion(_self, model_id="runwayml/stable-diffusion-v1-5"):
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if _self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(_self.device)
            return pipeline
        except Exception as e:
            st.warning(f"No se pudo cargar Stable Diffusion: {e}")
            return None

    def generate_precise_diffusion_prompt(self, card_data):
        """Genera prompt que coincide EXACTAMENTE con la descripción del usuario"""
        character = card_data.get('Character', 'Guerrero').lower()
        elements = card_data.get('Elements', [])
        original_prompt = card_data.get('original_prompt', '').lower()
        
        # Construir prompt específico basado en personaje
        if character == 'golem':
            main_subject = "massive stone golem creature, rock giant warrior, elemental golem"
        elif character == 'gigante':
            main_subject = "giant warrior, towering giant fighter, massive giant"
        elif character == 'caballero':
            main_subject = "armored knight warrior, medieval knight champion"
        elif character == 'dragón':
            main_subject = "majestic dragon creature, flying dragon beast, fantasy dragon"
        elif character == 'arquero':
            main_subject = "skilled archer warrior, bow-wielding archer, archer champion"
        elif character == 'mago':
            main_subject = "magical wizard sorcerer, staff-wielding wizard, arcane mage"
        elif any(word in character for word in ['hechizo', 'rayo', 'veneno']):
            main_subject = "magical spell effect, mystical energy projectile, spell casting magic"
        elif any(word in character for word in ['torre', 'cañón']):
            main_subject = "defensive tower building, cannon tower, medieval fortress"
        else:
            main_subject = f"medieval fantasy warrior, {character} fighter"
        
        # Añadir elementos específicos
        element_effects = []
        if 'hielo' in elements:
            element_effects.extend(["ice crystals", "frozen effects", "blue frost aura", "winter magic"])
        if 'fuego' in elements:
            element_effects.extend(["flames", "fire effects", "orange burning", "blazing aura"])
        if 'electrico' in elements:
            element_effects.extend(["lightning bolts", "electrical energy", "yellow sparks"])
        if 'veneno' in elements:
            element_effects.extend(["poison effects", "toxic aura", "green poison"])
        
        # Prompt final
        final_prompt = main_subject
        if element_effects:
            final_prompt += f", {', '.join(element_effects)}"
        final_prompt += ", Clash Royale game art style, Supercell design, detailed character design"
        
        return {
            'prompt': final_prompt,
            'negative_prompt': "realistic photo, blurry, low quality, text, watermark, modern clothing",
            'generation_params': {
                'num_inference_steps': 35,
                'guidance_scale': 8.0,
                'width': 512,
                'height': 640
            }
        }

    def generate_image_with_diffusion(self, prompt_data, pipeline):
        """Genera imagen con Stable Diffusion"""
        try:
            params = prompt_data['generation_params']
            prompt_hash = hashlib.md5(prompt_data['prompt'].encode()).hexdigest()
            seed = int(prompt_hash[:8], 16) % 1000000
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt_data['prompt'],
                    negative_prompt=prompt_data['negative_prompt'],
                    num_inference_steps=params['num_inference_steps'],
                    guidance_scale=params['guidance_scale'],
                    width=params['width'],
                    height=params['height'],
                    generator=generator
                )
            
            return result.images[0]
        except Exception as e:
            st.error(f"Error generando imagen: {e}")
            return None

    def create_card_composition(self, image, card_data, card_name):
        """Crea composición de carta con duración para hechizos"""
        card_width, card_height = 600, 800
        canvas = Image.new('RGB', (card_width, card_height), color='#1a1a1a')
        draw = ImageDraw.Draw(canvas)
        
        cost = int(card_data.get('Cost', 3))
        damage = int(card_data.get('Damage', 100))
        health = int(card_data.get('Health (+Shield)', 100))
        duration = int(card_data.get('Duration', 0))
        card_type = card_data.get('Type', 'Troops and Defenses')
        
        # Colores por rareza
        if cost <= 2:
            scheme = {'border': '#C0C0C0', 'bg': '#E8E8E8', 'rarity': 'COMÚN', 'cost_bg': '#4A90E2'}
        elif cost <= 4:
            scheme = {'border': '#FF8C00', 'bg': '#FFE4B5', 'rarity': 'RARO', 'cost_bg': '#FF6B35'}
        elif cost <= 6:
            scheme = {'border': '#9932CC', 'bg': '#E6E6FA', 'rarity': 'ÉPICO', 'cost_bg': '#8E44AD'}
        else:
            scheme = {'border': '#FFD700', 'bg': '#FFF8DC', 'rarity': 'LEGENDARIO', 'cost_bg': '#F39C12'}
        
        # Fondo principal
        draw.rounded_rectangle(
            [(15, 15), (card_width-15, card_height-15)],
            radius=25, fill=scheme['bg'], outline=scheme['border'], width=6
        )
        
        # Círculo de coste
        self._draw_cost_circle(draw, cost, scheme)
        
        # Área de imagen
        if image:
            self._draw_image_area(draw, card_width, image, scheme)
        
        # Información completa de la carta
        self._draw_card_info_complete(draw, card_width, card_height, card_data, card_name, scheme)
        
        return canvas
    
    def _draw_cost_circle(self, draw, cost, scheme):
        """Dibuja círculo de coste"""
        cost_center = (75, 75)
        cost_radius = 40
        
        draw.ellipse([
            cost_center[0]-cost_radius-4, cost_center[1]-cost_radius-4,
            cost_center[0]+cost_radius+4, cost_center[1]+cost_radius+4
        ], fill='white', outline=scheme['border'], width=3)
        draw.ellipse([
            cost_center[0]-cost_radius, cost_center[1]-cost_radius,
            cost_center[0]+cost_radius, cost_center[1]+cost_radius
        ], fill=scheme['cost_bg'], outline='white', width=2)
        
        try:
            font_cost = ImageFont.truetype("arial.ttf", 32)
        except:
            font_cost = ImageFont.load_default()
        
        cost_text = str(cost)
        bbox = draw.textbbox((0, 0), cost_text, font=font_cost)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((cost_center[0] - text_width//2, cost_center[1] - text_height//2), 
                 cost_text, fill='white', font=font_cost)
    
    def _draw_image_area(self, draw, width, image, scheme):
        """Dibuja área de imagen"""
        image_area = [(40, 120), (width-40, 420)]
        draw.rounded_rectangle(image_area, radius=20, fill='#2C3E50', outline=scheme['border'], width=4)
        
        if image:
            image_resized = image.resize((520, 290), Image.Resampling.LANCZOS)
            return (45, 125, image_resized)
        return None
    
    def _draw_card_info_complete(self, draw, width, height, card_data, card_name, scheme):
        """Dibuja información completa incluyendo duración para hechizos"""
        try:
            font_title = ImageFont.truetype("arial.ttf", 18)
            font_stats = ImageFont.truetype("arial.ttf", 16)
            font_desc = ImageFont.truetype("arial.ttf", 11)
        except:
            font_title = font_stats = font_desc = ImageFont.load_default()
        
        # Área de información
        info_area = [(40, 440), (width-40, height-20)]
        draw.rounded_rectangle(info_area, radius=15, fill='#34495E', outline=scheme['border'], width=3)
        
        # Título y rareza
        title_y = 460
        draw.text((60, title_y), card_name, fill='white', font=font_title)
        draw.text((width-120, title_y), scheme['rarity'], fill=scheme['border'], font=font_stats)
        
        # Tipo de carta
        type_y = title_y + 25
        type_mapping = {
            'Troops and Defenses': 'TROPA',
            'Damaging Spells': 'HECHIZO',
            'Spawners': 'EDIFICIO'
        }
        card_type_display = type_mapping.get(card_data['Type'], 'TROPA')
        draw.text((60, type_y), card_type_display, fill='#3498DB', font=font_stats)
        
        # Estadísticas con duración 
        stats_y = 510
        self._draw_card_stats_with_exact_duration(draw, card_data, stats_y, font_stats)
        
        # Narrativa completa
        desc_y = stats_y + 35
        narrative = card_data.get('Narrative', '')
        self._draw_complete_text(draw, narrative, 60, desc_y, width-120, font_desc, '#BDC3C7')
    
    def _draw_card_stats_with_exact_duration(self, draw, card_data, y, font):
        """Dibuja estadísticas incluyendo duración para hechizos"""
        card_type = card_data.get('Type', 'Troops and Defenses')
        duration = int(card_data.get('Duration', 0))
        
        if card_type == 'Damaging Spells':
            # Para hechizos: DAÑO y DURACIÓN 
            draw.text((60, y), "DAÑO", fill='#E74C3C', font=font)
            draw.text((150, y), str(card_data.get('Damage', 0)), fill='white', font=font)
            draw.text((250, y), "DURACIÓN", fill='#F39C12', font=font)
            if duration == 1:
                draw.text((360, y), "Instant.", fill='white', font=font)
            else:
                draw.text((360, y), f"{duration}s", fill='white', font=font)
        elif card_type == 'Spawners':
            draw.text((60, y), "VIDA", fill='#27AE60', font=font)
            draw.text((150, y), str(card_data.get('Health (+Shield)', 0)), fill='white', font=font)
        else:
            draw.text((60, y), "DAÑO", fill='#E74C3C', font=font)
            draw.text((140, y), str(card_data.get('Damage', 0)), fill='white', font=font)
            draw.text((220, y), "VIDA", fill='#27AE60', font=font)
            draw.text((300, y), str(card_data.get('Health (+Shield)', 0)), fill='white', font=font)
    
    def _draw_complete_text(self, draw, text, x, y, max_width, font, color):
        """Dibuja texto completo con wrapping mejorado"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " " if current_line else word + " "
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width - 20:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        # Dibujar todas las líneas
        line_height = 15
        max_lines = min(len(lines), 8)
        
        for i, line in enumerate(lines[:max_lines]):
            draw.text((x, y + i * line_height), line, fill=color, font=font)
        
        if len(lines) > max_lines:
            draw.text((x, y + max_lines * line_height), "...", fill=color, font=font)

def main():
    # Header
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF6B35;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
    <div class="main-header">Generador de Cartas Clash Royale IA</div>
    <div class="subtitle">Crea tu carta como la imagines ;)</div>
    """, unsafe_allow_html=True)
    
    # Sidebar con instrucciones
    with st.sidebar:
        st.header("📋 Instrucciones")
        st.markdown("""
        **✅ SÉ MUY ESPECÍFICO CON NÚMEROS:**
        - "Golem de hielo con 400 de daño y 200 de vida"
        - "Hechizo que cueste 4 elixir y cause 300 de daño"
        - "Hechizo que dure 4 segundos" (para hechizos)
        
        **🎯 MENCIONA EL PERSONAJE:**
        - Golem, Gigante, Caballero, Dragón
        - Arquero, Mago, Torre, Cañón
        - Hechizo, Rayo, Veneno, Ralentizar
        
        **🔥 AÑADE ELEMENTOS:**
        - "de hielo", "de fuego", "eléctrico"
        - "que congele", "que queme", "que ralentice"
        
        **⏱️ PARA HECHIZOS - DURACIÓN:**
        - "que dure 5 segundos"
        - "efecto de 3 segundos"
        - "por 8 segundos"
        """)
    
    # Área principal
    st.subheader("🎯 Describe tu carta con números específicos")
    user_prompt = st.text_area(
        "Descripción de tu carta:",
        height=120,
        placeholder="Ejemplo: 'Golem de hielo con 400 de daño y 200 de vida' o 'Hechizo de fuego que cueste 4 elixir, cause 350 de daño y dure 4 segundos'"
    )
    
    # Botón principal
    if st.button("Generar Carta", type="primary", use_container_width=True):
        if not user_prompt.strip():
            st.warning("Por favor, describe la carta que deseas crear con números específicos.")
            return
        
        with st.spinner("Analizando tu descripción exacta..."):
            # Parser que RESPETA el prompt
            card_data = parse_user_prompt_precisely(user_prompt)
        
        with st.spinner("Generando nombre y narrativa precisos..."):
            # Generar nombre y narrativa que coinciden con el prompt
            card_name = generate_precise_card_name(card_data)
            narrative = generate_precise_narrative(card_data)
            card_data['Narrative'] = narrative
        
        # Mostrar resultados
        st.success("¡Carta generada exitosamente")
        
        # Información de la carta
        if card_data['Type'] == 'Damaging Spells':
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💎 Coste", f"{card_data['Cost']} elixir")
            with col2:
                st.metric("⚔️ Daño", card_data['Damage'])
            with col3:
                st.metric("🛡️ Vida", card_data['Health (+Shield)'])
            with col4:
                duration = card_data['Duration']
                if duration == 1:
                    st.metric("⏱️ Duración", "Instant.")
                else:
                    st.metric("⏱️ Duración", f"{duration}s")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💎 Coste", f"{card_data['Cost']} elixir")
                st.metric("⚔️ Daño", card_data['Damage'])
            with col2:
                st.metric("🛡️ Vida", card_data['Health (+Shield)'])
                type_display = {
                    'Troops and Defenses': '🛡️ Tropa',
                    'Damaging Spells': '⚡ Hechizo',
                    'Spawners': '🏗️ Edificio'
                }
                st.metric("🎭 Tipo", type_display.get(card_data['Type'], '🛡️ Tropa'))
            with col3:
                cost = card_data['Cost']
                rarity = "⚪ COMÚN" if cost <= 2 else "🟠 RARO" if cost <= 4 else "🟣 ÉPICO" if cost <= 6 else "🟡 LEGENDARIO"
                st.metric("⭐ Rareza", rarity)
        
        # Nombre y narrativa
        st.subheader(f"🏆 {card_name}")
        st.info(card_data['Narrative'])
        
        # Generación de imagen
        generator = StableDiffusionCardGenerator()
        
        with st.spinner("🎨 Cargando modelo de generación..."):
            pipeline = generator.setup_stable_diffusion()
        
        if pipeline:
            with st.spinner("🖼️ Creando imagen que coincide con tu descripción..."):
                prompt_data = generator.generate_precise_diffusion_prompt(card_data)
                image = generator.generate_image_with_diffusion(prompt_data, pipeline)
                
                if image:
                    with st.spinner("🎴 Componiendo carta final..."):
                        card_img = generator.create_card_composition(image, card_data, card_name)
                        # Pegar la imagen
                        image_info = generator._draw_image_area(ImageDraw.Draw(card_img), 600, image, {'border': '#FFD700'})
                        if image_info:
                            card_img.paste(image_info[2], (image_info[0], image_info[1]))
                    
                    st.subheader("🏆 Tu carta personalizada")
                    st.image(card_img, caption=f"Carta: {card_name}", use_container_width=True)
                    
                    # Descarga
                    import io
                    img_buffer = io.BytesIO()
                    card_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="Descargar carta completa",
                        data=img_buffer.getvalue(),
                        file_name=f"carta_{card_name.replace(' ', '_').lower()}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Error generando la imagen.")
        else:
            st.warning("⚠️ No se pudo cargar el modelo de imágenes.")

if __name__ == "__main__":
    main()


