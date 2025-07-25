import streamlit as st
import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import random
import re

# ========== CLASES Y FUNCIONES DEL DIFUSOR ==========

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class StableDiffusionCardGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.style_templates = {
            'Troops and Defenses': {
                'visual_style': 'fantasy medieval warrior, armored knight, battle-ready soldier',
                'color_palette': ['golden armor', 'steel blue', 'crimson red', 'forest green'],
                'environment': 'medieval battlefield, castle courtyard, arena combat',
                'mood': 'heroic, powerful, determined'
            },
            'Damaging Spells': {
                'visual_style': 'magical energy, spell casting, mystical explosion, arcane power',
                'color_palette': ['electric blue', 'fiery orange', 'purple magic', 'lightning yellow'],
                'environment': 'magical arena, energy vortex, spell circle, mystical sky',
                'mood': 'destructive, magical, intense'
            },
            'Spawners': {
                'visual_style': 'magical tower, fantasy building, troop generator, mystical structure',
                'color_palette': ['stone gray', 'mystical purple', 'emerald green', 'ancient gold'],
                'environment': 'fantasy architecture, magical base, enchanted fortress',
                'mood': 'mysterious, powerful, ancient'
            }
        }

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

    def generate_diffusion_prompt(self, card_data):
        cost = int(card_data.get('Cost', 3))
        damage = int(card_data.get('Damage', 100))
        health = int(card_data.get('Health (+Shield)', 100))
        card_type = card_data.get('Type', 'Troops and Defenses')
        narrative = card_data.get('Narrative', '')
        is_ice_spell = ('nieve' in narrative.lower() or 'hielo' in narrative.lower() or
                        'frozen' in narrative.lower() or damage < 150)
        if card_type == 'Damaging Spells':
            if is_ice_spell:
                main_subject = "large magical snowball projectile, giant ice orb spell, frozen magic sphere, swirling snow particles, winter magic effect, crystalline ice surface, magical frost aura, blue and white ice magic, frozen spell projectile, Clash Royale ice spell style, magical ice attack"
            elif damage > 400:
                main_subject = "massive magical explosion, devastating fire spell, huge energy blast, fiery magical eruption, destructive spell effect, orange and red flames, powerful magic destruction, Clash Royale fireball style"
            else:
                main_subject = "medium magical projectile, energy spell casting, colorful magic bolt, spell energy effect, magical attack, Clash Royale arrow style"
        elif card_type == 'Spawners':
            main_subject = "medieval defensive tower, fantasy castle building, stone fortress structure, magical spawning building, defensive architecture, Clash Royale building style, military tower with magical aura, spawner building design"
        else:  # Tropas
            if damage > 300 and health > 1000:
                main_subject = "heavily armored medieval knight, massive warrior with full plate armor, legendary champion fighter, imposing armored hero, Clash Royale prince style, golden armor with blue details, heroic knight character"
            elif damage > 300:
                main_subject = "fierce medieval warrior, battle-ready armored fighter, aggressive knight, sharp weapon wielding soldier, Clash Royale mini pekka style, dark armor with menacing appearance"
            elif health > 1000:
                main_subject = "massive giant warrior, enormous friendly fighter, huge armored defender, oversized gentle giant, Clash Royale giant style, protective tank character"
            else:
                main_subject = "balanced medieval knight, versatile armored warrior, classic knight design, reliable fighter with sword and armor, Clash Royale knight style, blue and gold armor color scheme"
        enhanced_prompt = f"{main_subject}, Clash Royale mobile game art style, Supercell game design, cartoonish 3D rendered character, bright vibrant colors, high contrast lighting, dynamic heroic pose, professional mobile game artwork, detailed game illustration, arena battlefield background, fantasy medieval style, high quality game art"
        return {
            'prompt': enhanced_prompt,
            'negative_prompt': "realistic photo, photorealistic, dark lighting, scary, horror, modern clothing, low quality, blurry, distorted, text, watermark, amateur art, sketch",
            'generation_params': {
                'num_inference_steps': 40 if cost >= 6 else 30,
                'guidance_scale': 9.5 if cost >= 6 else 8.5,
                'width': 512,
                'height': 640
            }
        }

    def generate_image_with_diffusion(self, prompt_data, card_id, pipeline):
        try:
            params = prompt_data['generation_params']
            generator = torch.Generator(device=self.device).manual_seed(42 + card_id)
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt_data['prompt'],
                    negative_prompt=prompt_data['negative_prompt'],
                    num_inference_steps=params['num_inference_steps'],
                    guidance_scale=params['guidance_scale'],
                    width=params['width'],
                    height=params['height'],
                    num_images_per_prompt=1,
                    generator=generator
                )
            image = result.images[0]
            return image
        except Exception as e:
            st.warning(f"Error generando imagen: {e}")
            return None

    def create_card_composition(self, image, card_data, card_id):
        # Composición visual de carta
        card_width, card_height = 600, 800
        canvas = Image.new('RGB', (card_width, card_height), color='#1a1a1a')
        draw = ImageDraw.Draw(canvas)
        cost = int(card_data.get('Cost', 3))
        damage = int(card_data.get('Damage', 100))
        health = int(card_data.get('Health (+Shield)', 100))
        card_type = card_data.get('Type', 'Troops and Defenses')
        # Colores y rareza
        if cost <= 2:
            border_color = '#C0C0C0'; bg_color = '#E8E8E8'; rarity_text = "COMÚN"; cost_bg = '#4A90E2'
        elif cost <= 4:
            border_color = '#FF8C00'; bg_color = '#FFE4B5'; rarity_text = "RARO"; cost_bg = '#FF6B35'
        elif cost <= 6:
            border_color = '#9932CC'; bg_color = '#E6E6FA'; rarity_text = "ÉPICO"; cost_bg = '#8E44AD'
        else:
            border_color = '#FFD700'; bg_color = '#FFF8DC'; rarity_text = "LEGENDARIO"; cost_bg = '#F39C12'
        # Fondo principal
        draw.rounded_rectangle([(15, 15), (card_width-15, card_height-15)], radius=25, fill=bg_color, outline=border_color, width=6)
        # Círculo de coste
        cost_center = (75, 75); cost_radius = 40
        draw.ellipse([cost_center[0]-cost_radius-4, cost_center[1]-cost_radius-4, cost_center[0]+cost_radius+4, cost_center[1]+cost_radius+4], fill='white', outline=border_color, width=3)
        draw.ellipse([cost_center[0]-cost_radius, cost_center[1]-cost_radius, cost_center[0]+cost_radius, cost_center[1]+cost_radius], fill=cost_bg, outline='white', width=2)
        # Área de imagen
        image_area = [(40, 120), (card_width-40, 420)]
        draw.rounded_rectangle(image_area, radius=20, fill='#2C3E50', outline=border_color, width=4)
        if image:
            image_resized = image.resize((520, 290), Image.Resampling.LANCZOS)
            canvas.paste(image_resized, (45, 125))
        # Fuentes
        try:
            from PIL import ImageFont
            font_cost = ImageFont.truetype("arial.ttf", 32)
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_stats = ImageFont.truetype("arial.ttf", 16)
            font_desc = ImageFont.truetype("arial.ttf", 12)
        except:
            font_cost = font_title = font_stats = font_desc = None
        # Coste
        cost_text = str(cost)
        draw.text((cost_center[0]-10, cost_center[1]-15), cost_text, fill='white', font=font_cost)
        # Info área
        info_area = [(40, 440), (card_width-40, card_height-40)]
        draw.rounded_rectangle(info_area, radius=15, fill='#34495E', outline=border_color, width=3)
        # Título y rareza
        title_y = 460
        card_name = f"CARTA IA #{card_id:02d}"
        draw.text((60, title_y), card_name, fill='white', font=font_title)
        draw.text((card_width-120, title_y), rarity_text, fill=border_color, font=font_stats)
        # Tipo de carta
        type_y = title_y + 25
        type_mapping = {'Troops and Defenses': 'TROPA', 'Damaging Spells': 'HECHIZO', 'Spawners': 'EDIFICIO'}
        card_type_display = type_mapping.get(card_type, 'TROPA')
        draw.text((60, type_y), card_type_display, fill='#3498DB', font=font_stats)
        # Stats
        stats_y = 510
        if card_type == 'Damaging Spells':
            draw.text((60, stats_y), "DAÑO", fill='#E74C3C', font=font_stats)
            draw.text((150, stats_y), str(damage), fill='white', font=font_stats)
            draw.text((250, stats_y), "RADIO", fill='#F39C12', font=font_stats)
            draw.text((340, stats_y), "3.0", fill='white', font=font_stats)
            draw.text((420, stats_y), "RALENT.", fill='#3498DB', font=font_stats)
            draw.text((520, stats_y), "2.5s", fill='white', font=font_stats)
        elif card_type == 'Spawners':
            draw.text((60, stats_y), "VIDA", fill='#27AE60', font=font_stats)
            draw.text((150, stats_y), str(health), fill='white', font=font_stats)
            draw.text((250, stats_y), "DURACIÓN", fill='#9B59B6', font=font_stats)
            draw.text((360, stats_y), "60s", fill='white', font=font_stats)
        else:
            draw.text((60, stats_y), "DAÑO", fill='#E74C3C', font=font_stats)
            draw.text((140, stats_y), str(damage), fill='white', font=font_stats)
            draw.text((220, stats_y), "VIDA", fill='#27AE60', font=font_stats)
            draw.text((300, stats_y), str(health), fill='white', font=font_stats)
            draw.text((380, stats_y), "VEL.", fill='#F39C12', font=font_stats)
            draw.text((460, stats_y), "1.2s", fill='white', font=font_stats)
        # Descripción
        desc_y = stats_y + 40
        narrative = card_data.get('Narrative', '')
        words = narrative.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if len(test_line) <= 45:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        for i, line in enumerate(lines[:3]):
            draw.text((60, desc_y + i*18), line, fill='#BDC3C7', font=font_desc)
        footer_y = card_height - 70
        draw.text((60, footer_y), f"Coste: {cost} elixir", fill='#F39C12', font=font_desc)
        draw.text((300, footer_y), f"Generado por IA", fill='#95A5A6', font=font_desc)
        level_area = (card_width-80, card_height-50, card_width-20, card_height-20)
        draw.rounded_rectangle(level_area, radius=5, fill='#2C3E50', outline=border_color, width=2)
        draw.text((card_width-60, card_height-40), "9", fill='white', font=font_stats)
        return canvas

# ========== FUNCIONES DE NARRATIVA Y PARSEO (del backend) ==========

def clean_spanish_text_advanced(text):
    english_words = [
        'vernal', 'recommended', 'spawner', 'locations', 'troops', 'defenses',
        'damage', 'health', 'attack', 'defense', 'spell', 'building', 'unit'
    ]
    for word in english_words:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s\.\!\?\,\:\;áéíóúÁÉÍÓÚñÑüÜ]', '', text)
    text = re.sub(r'\?+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_premium_clash_narrative(attributes):
    cost = int(attributes.get('Cost', 3))
    damage = int(attributes.get('Damage', 0))
    health = int(attributes.get('Health (+Shield)', 0))
    card_type = attributes.get('Type', 'Troops and Defenses')
    if card_type == 'Damaging Spells':
        if cost <= 2:
            templates = [
                f"¡Hechizo rápido de {cost} elixir! Causa {damage} de daño instantáneo. Perfecto para eliminar tropas pequeñas y sorprender al enemigo.",
                f"¡Magia económica de {cost} elixir! Inflige {damage} puntos de daño en área. Los rivales no verán venir este devastador conjuro.",
                f"¡Conjuro veloz de {cost} elixir! Destruye con {damage} de daño. Ideal para ciclos rápidos y ataques sorpresa definitivos."
            ]
        elif cost <= 4:
            templates = [
                f"¡Hechizo poderoso de {cost} elixir! Arrasa enemigos con {damage} de daño brutal en zona amplia. Cambiará el destino de la batalla.",
                f"¡Conjuro versátil de {cost} elixir! Aniquila rivales con {damage} puntos de daño devastador. Úsalo sabiamente para la victoria.",
                f"¡Magia destructiva de {cost} elixir! Causa {damage} de daño letal en área. Los enemigos huirán aterrorizados del campo."
            ]
        else:
            templates = [
                f"¡Hechizo legendario de {cost} elixir! Aniquila todo con {damage} de daño masivo apocalíptico. Devastación total garantizada en la arena.",
                f"¡Conjuro supremo de {cost} elixir! Destrucción absoluta de {damage} puntos letales. Dominará completamente toda la arena de batalla.",
                f"¡Magia definitiva de {cost} elixir! Poder destructivo de {damage}. La victoria está completamente asegurada para siempre."
            ]
    elif card_type == 'Spawners':
        if cost <= 4:
            templates = [
                f"¡Torre productora de {cost} elixir! Genera tropas continuamente sin parar. Resistencia sólida para defender tu corona victoriosamente.",
                f"¡Edificio spawner de {cost} elixir! Invoca unidades automáticamente en oleadas. Presión constante que abrumará a los enemigos.",
                f"¡Estructura generadora de {cost} elixir! Produce ejércitos sin descanso. Los rivales no podrán avanzar ni un solo paso."
            ]
        else:
            templates = [
                f"¡Mega fortaleza de {cost} elixir! Genera oleadas masivas de tropas imparables. Dominará completamente el campo de batalla enemigo.",
                f"¡Super edificio de {cost} elixir! Invoca ejércitos legendarios continuamente. La presión será absolutamente abrumadora para los rivales.",
                f"¡Torre suprema de {cost} elixir! Producción masiva garantizada eternamente. Los enemigos se rendirán antes de la primera oleada."
            ]
    else: # Tropas
        if cost <= 2:
            if damage > 100:
                templates = [
                    f"¡Guerrero feroz de {cost} elixir! Ataque brutal de {damage} por golpe mortal. Perfecto para ataques sorpresa devastadores.",
                    f"¡Luchador veloz de {cost} elixir! Golpea con {damage} de daño letal. Los enemigos caerán antes de reaccionar.",
                    f"¡Asesino rápido de {cost} elixir! Causa {damage} puntos por impacto. Ideal para ciclos de muerte imparables."
                ]
            else:
                templates = [
                    f"¡Tropa económica de {cost} elixir! Resistencia sólida de {health} puntos. Perfecta para distraer y confundir enemigos.",
                    f"¡Unidad barata de {cost} elixir! Aguanta {health} de daño heroicamente. Excelente para defensa y contraataques.",
                    f"¡Soldado accesible de {cost} elixir! Vida resistente de {health}. Los rivales gastarán elixir extra innecesariamente."
                ]
        elif cost <= 4:
            if damage > 300:
                templates = [
                    f"¡Guerrero implacable de {cost} elixir! Daño devastador de {damage} por golpe brutal. Arrasará con cualquier enemigo del camino.",
                    f"¡Luchador legendario de {cost} elixir! Ataque mortal de {damage} puntos. Los rivales huirán aterrorizados de su poder.",
                    f"¡Soldado feroz de {cost} elixir! Golpe letal de {damage}. Devastación pura que aniquilará toda resistencia enemiga."
                ]
            elif health > 1500:
                templates = [
                    f"¡Tanque invencible de {cost} elixir! Vida masiva de {health} puntos épicos. Absorbe todo el daño enemigo sin inmutarse.",
                    f"¡Muro viviente de {cost} elixir! Resistencia titanica de {health}. Ningún ataque enemigo podrá detener su avance.",
                    f"¡Fortaleza móvil de {cost} elixir! Aguanta {health} de daño heroicamente. Los enemigos se cansarán de atacar inútilmente."
                ]
            else:
                templates = [
                    f"¡Tropa equilibrada de {cost} elixir! Combina {damage} de ataque y {health} de vida perfectamente. Versatilidad total asegurada.",
                    f"¡Unidad completa de {cost} elixir! Estadísticas balanceadas ideales para toda situación. Funcionará en cualquier estrategia.",
                    f"¡Soldado versátil de {cost} elixir! Poder y resistencia combinados magistralmente. Perfecto para cualquier táctica de batalla."
                ]
        else:
            if damage > 500:
                templates = [
                    f"¡Bestia legendaria de {cost} elixir! Poder destructivo de {damage} apocalíptico. Aniquilará completamente cualquier ejército enemigo existente.",
                    f"¡Titán imparable de {cost} elixir! Fuerza brutal de {damage} devastadora. Los rivales abandonarán la partida al verlo aparecer.",
                    f"¡Monstruo definitivo de {cost} elixir! Daño letal de {damage}. Dominación total asegurada para toda la eternidad."
                ]
            elif health > 3000:
                templates = [
                    f"¡Coloso invencible de {cost} elixir! Resistencia épica de {health} puntos legendarios. Será completamente imposible de destruir.",
                    f"¡Gigante supremo de {cost} elixir! Vida masiva de {health}. Absorbe cualquier ataque sin sufrir daño significativo.",
                    f"¡Titán defensivo de {cost} elixir! Aguanta {health} de daño épico. Los enemigos se agotarán antes de derrotarlo."
                ]
            else:
                templates = [
                    f"¡Campeón premium de {cost} elixir! Estadísticas superiores balanceadas perfectamente. Dominará toda la arena con superioridad.",
                    f"¡Unidad élite de {cost} elixir! Poder y resistencia combinados magistralmente. La victoria está completamente garantizada.",
                    f"¡Guerrero supremo de {cost} elixir! Perfección absoluta en combate. Los rivales no tienen ni la menor oportunidad."
                ]
    selected_template = random.choice(templates)
    clean_narrative = clean_spanish_text_advanced(selected_template)
    if len(clean_narrative.split()) < 8:
        fallback = f"¡Carta poderosa de {cost} elixir! Perfecta para dominar la arena. Los enemigos temerán su increíble poder destructivo."
        return clean_spanish_text_advanced(fallback)
    return clean_narrative

def parse_user_prompt(prompt):
    cost = None
    damage = None
    health = None
    card_type = None
    if "hechizo" in prompt.lower() or "spell" in prompt.lower():
        card_type = "Damaging Spells"
    elif "edificio" in prompt.lower() or "spawner" in prompt.lower():
        card_type = "Spawners"
    else:
        card_type = "Troops and Defenses"
    numbers = [int(s) for s in re.findall(r'\d+', prompt)]
    if len(numbers) >= 3:
        cost, damage, health = numbers[:3]
    else:
        cost = random.randint(1, 8)
        damage = random.randint(50, 800)
        health = random.randint(100, 3000)
    return {
        'Cost': cost,
        'Damage': damage,
        'Health (+Shield)': health,
        'Type': card_type
    }

# ========== STREAMLIT APP ==========

st.title("Generador de Cartas Clash Royale")
st.write("Describe la carta que deseas (tipo, coste, daño, salud, etc). Ejemplo: 'Hechizo ralentizador que cueste 4 de elixir y haga 300 de daño'")

user_prompt = st.text_area("Descripción de la carta", height=80)

if st.button("Generar carta"):
    atributos = parse_user_prompt(user_prompt)
    narrativa = generate_premium_clash_narrative(atributos)
    card_data = atributos.copy()
    card_data['Narrative'] = narrativa

    st.subheader("Narrativa generada:")
    st.write(f"**Tipo:** {card_data['Type']}")
    st.write(f"**Coste:** {card_data['Cost']} elixir")
    st.write(f"**Daño:** {card_data['Damage']}")
    st.write(f"**Salud:** {card_data['Health (+Shield)']}")
    st.write(f"**Narrativa:** {card_data['Narrative']}")

    # Generador visual
    generator = StableDiffusionCardGenerator()
    pipeline = generator.setup_stable_diffusion()
    if pipeline is not None:
        prompt_data = generator.generate_diffusion_prompt(card_data)
        image = generator.generate_image_with_diffusion(prompt_data, 1, pipeline)
        card_img = generator.create_card_composition(image, card_data, 1)
        st.image(card_img, caption="Carta generada", use_container_width=True)
    else:
        st.warning("No se pudo cargar el modelo de difusión. Se mostrará solo la narrativa.")