from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.switch import Switch

import cv2
import numpy as np
import pytesseract
import sqlite3
import os
import re
import threading
import time
import datetime
import webbrowser
import urllib.parse
from functools import partial

# Configuración para dispositivos móviles
# En Android, necesitarás configurar la ruta de Tesseract en el archivo .buildozer/android/app/sitecustomize.py
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class LogMessage:
    def __init__(self, text, level="info"):
        self.text = text
        self.level = level
        self.timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    def __str__(self):
        return f"[{self.timestamp}] {self.text}"

class PlateRecognitionApp(App):
    title = "Sistema de Reconocimiento de Placas - Bolivia"
    
    def build(self):
        # Crear la interfaz principal
        self.main_layout = MainLayout()
        
        # Inicializar la base de datos
        self.create_sample_database()
        
        return self.main_layout
    
    def create_sample_database(self):
        """Crea una base de datos de ejemplo con vehículos y propietarios"""
        conn = sqlite3.connect('parking_db.sqlite')
        cursor = conn.cursor()
        
        # Crear tablas si no existen
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY,
            plate TEXT UNIQUE,
            model TEXT,
            owner_name TEXT,
            phone TEXT,
            parking_spot TEXT
        )
        ''')
        
        # Insertar datos de ejemplo con formato de placas bolivianas
        sample_data = [
            ('1234ABC', 'Toyota Corolla', 'Juan Pérez', '+59171234567', 'A-15'),
            ('5678DEF', 'Honda Civic', 'María García', '+59173456789', 'B-22'),
            ('9012GHI', 'Ford Focus', 'Carlos López', '+59175678901', 'C-07'),
            ('3456JKL', 'Volkswagen Golf', 'Ana Martínez', '+59177890123', 'D-04'),
            ('7890MNO', 'Chevrolet Cruze', 'Roberto Sánchez', '+59179012345', 'E-19'),
            ('6345AIT', 'KIA Carens', 'Jhon Calsina', '59177232382', 'E-20'),
            ('3529TBD', 'Suzuki Fronx', 'Alex Medina', '59177232382', 'E-19'),
            ('1852PHD', 'Hyundai Santa Fe', 'Ricardo Gabriel', '59177232382', 'E-18')
        ]
        
        for vehicle in sample_data:
            try:
                cursor.execute('''
                INSERT OR IGNORE INTO vehicles (plate, model, owner_name, phone, parking_spot)
                VALUES (?, ?, ?, ?, ?)
                ''', vehicle)
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
        
        self.main_layout.add_log("Base de datos inicializada con placas bolivianas")

class MainLayout(BoxLayout):
    # Variables para almacenar la información del vehículo
    plate_text = StringProperty("")
    vehicle_text = StringProperty("")
    owner_text = StringProperty("")
    spot_text = StringProperty("")
    phone_text = StringProperty("")
    
    # Estado de la cámara
    is_camera_active = BooleanProperty(False)
    
    # Intervalo de reconocimiento (en segundos) - Reducido para mayor velocidad
    recognition_interval = 0.5
    
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.padding = 10
        self.spacing = 10
        
        # Variables para la cámara y el procesamiento
        self.capture = None
        self.current_image = None
        self.recognition_event = None
        self.last_recognized_plate = None
        self.last_recognition_time = 0
        self.cooldown_period = 3  # Reducido para mayor respuesta
        
        # Opciones de procesamiento
        self.use_low_resolution = True  # Usar resolución baja para procesamiento más rápido
        self.use_fast_detection = True  # Usar detección rápida
        
        # Crear los paneles izquierdo y derecho
        self.create_left_panel()
        self.create_right_panel()
        
        # Inicializar logs
        self.logs = []
        self.add_log("Sistema iniciado - Optimizado para placas bolivianas")
    
    def create_left_panel(self):
        """Crea el panel izquierdo con la cámara y controles"""
        left_panel = BoxLayout(orientation='vertical', size_hint=(0.6, 1))
        
        # Área de la cámara
        self.camera_image = Image(size_hint=(1, 0.7))
        left_panel.add_widget(self.camera_image)
        
        # Controles de la cámara
        controls = BoxLayout(size_hint=(1, 0.15), spacing=10, padding=10)
        
        self.camera_button = Button(
            text="Iniciar Cámara", 
            on_press=self.toggle_camera,
            background_color=(0.2, 0.7, 0.3, 1)
        )
        controls.add_widget(self.camera_button)
        
        self.recognition_button = Button(
            text="Reconocimiento: OFF",
            on_press=self.toggle_auto_recognition,
            background_color=(0.7, 0.3, 0.3, 1)
        )
        controls.add_widget(self.recognition_button)
        
        left_panel.add_widget(controls)
        
        # Opciones de optimización
        options = BoxLayout(size_hint=(1, 0.15), spacing=10, padding=10)
        
        # Opción de resolución baja
        options.add_widget(Label(text="Modo rápido:"))
        self.fast_mode_switch = Switch(active=True)
        self.fast_mode_switch.bind(active=self.toggle_fast_mode)
        options.add_widget(self.fast_mode_switch)
        
        # Botón de captura manual
        self.capture_button = Button(
            text="Capturar Ahora", 
            on_press=self.perform_recognition,
            background_color=(0.3, 0.3, 0.7, 1)
        )
        options.add_widget(self.capture_button)
        
        left_panel.add_widget(options)
        
        self.add_widget(left_panel)
    
    def create_right_panel(self):
        """Crea el panel derecho con la información del vehículo y logs"""
        right_panel = BoxLayout(orientation='vertical', size_hint=(0.4, 1))
        
        # Información del vehículo
        info_panel = GridLayout(cols=2, size_hint=(1, 0.5), spacing=5, padding=10)
        
        # Etiquetas y campos de texto
        info_panel.add_widget(Label(text="Placa:"))
        self.plate_input = TextInput(text=self.plate_text, readonly=True)
        info_panel.add_widget(self.plate_input)
        
        info_panel.add_widget(Label(text="Vehículo:"))
        self.vehicle_input = TextInput(text=self.vehicle_text, readonly=True)
        info_panel.add_widget(self.vehicle_input)
        
        info_panel.add_widget(Label(text="Propietario:"))
        self.owner_input = TextInput(text=self.owner_text, readonly=True)
        info_panel.add_widget(self.owner_input)
        
        info_panel.add_widget(Label(text="Lugar:"))
        self.spot_input = TextInput(text=self.spot_text, readonly=True)
        info_panel.add_widget(self.spot_input)
        
        info_panel.add_widget(Label(text="Teléfono:"))
        self.phone_input = TextInput(text=self.phone_text, readonly=True)
        info_panel.add_widget(self.phone_input)
        
        right_panel.add_widget(info_panel)
        
        # Botones de acción
        action_layout = BoxLayout(size_hint=(1, 0.15), spacing=10, padding=10)
        
        # Botón de WhatsApp
        self.whatsapp_button = Button(
            text="WhatsApp Directo",
            size_hint=(0.5, 1),
            background_color=(0.1, 0.7, 0.1, 1),
            on_press=self.send_whatsapp_direct
        )
        action_layout.add_widget(self.whatsapp_button)
        
        # Botón de WhatsApp Web
        self.whatsapp_web_button = Button(
            text="WhatsApp Web",
            size_hint=(0.5, 1),
            background_color=(0.1, 0.5, 0.1, 1),
            on_press=self.send_whatsapp_web
        )
        action_layout.add_widget(self.whatsapp_web_button)
        
        right_panel.add_widget(action_layout)
        
        # Área de logs
        log_label = Label(text="Registro de Actividad:", size_hint=(1, 0.1), halign='left')
        log_label.bind(size=log_label.setter('text_size'))
        right_panel.add_widget(log_label)
        
        scroll_view = ScrollView(size_hint=(1, 0.25))
        self.log_layout = GridLayout(cols=1, size_hint_y=None, spacing=2)
        self.log_layout.bind(minimum_height=self.log_layout.setter('height'))
        scroll_view.add_widget(self.log_layout)
        right_panel.add_widget(scroll_view)
        
        self.add_widget(right_panel)
    
    def toggle_fast_mode(self, instance, value):
        """Activa o desactiva el modo rápido"""
        self.use_low_resolution = value
        self.use_fast_detection = value
        self.add_log(f"Modo rápido: {'Activado' if value else 'Desactivado'}")
    
    def toggle_camera(self, instance):
        """Activa o desactiva la cámara"""
        if self.is_camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Inicia la cámara"""
        try:
            self.capture = cv2.VideoCapture(0)  # 0 es la cámara predeterminada
            
            if not self.capture.isOpened():
                self.add_log("Error: No se pudo abrir la cámara", "error")
                self.show_error("No se pudo acceder a la cámara")
                return
            
            # Configurar la cámara para resolución más baja (más rápido)
            if self.use_low_resolution:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_camera_active = True
            self.camera_button.text = "Detener Cámara"
            self.camera_button.background_color = (0.7, 0.3, 0.3, 1)
            
            # Iniciar actualización de frames
            self.camera_event = Clock.schedule_interval(self.update_camera, 1.0/30.0)  # 30 FPS
            self.add_log("Cámara iniciada")
            
        except Exception as e:
            self.add_log(f"Error al iniciar la cámara: {str(e)}", "error")
            self.show_error(f"Error al iniciar la cámara: {str(e)}")
    
    def stop_camera(self):
        """Detiene la cámara"""
        if self.camera_event:
            self.camera_event.cancel()
        
        if self.recognition_event:
            self.recognition_event.cancel()
            self.recognition_event = None
            self.recognition_button.text = "Reconocimiento: OFF"
            self.recognition_button.background_color = (0.7, 0.3, 0.3, 1)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        self.is_camera_active = False
        self.camera_button.text = "Iniciar Cámara"
        self.camera_button.background_color = (0.2, 0.7, 0.3, 1)
        self.add_log("Cámara detenida")
    
    def update_camera(self, dt):
        """Actualiza la imagen de la cámara"""
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                # Guardar el frame actual para procesamiento
                self.current_image = frame.copy()
                
                # Convertir el frame para mostrarlo en Kivy
                buf = cv2.flip(frame, 0)  # Voltear verticalmente
                buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                
                # Actualizar la imagen
                self.camera_image.texture = texture
    
    def toggle_auto_recognition(self, instance):
        """Activa o desactiva el reconocimiento automático"""
        if not self.is_camera_active:
            self.show_error("Primero debe iniciar la cámara")
            return
        
        if self.recognition_event:
            # Desactivar reconocimiento automático
            self.recognition_event.cancel()
            self.recognition_event = None
            self.recognition_button.text = "Reconocimiento: OFF"
            self.recognition_button.background_color = (0.7, 0.3, 0.3, 1)
            self.add_log("Reconocimiento automático desactivado")
        else:
            # Activar reconocimiento automático
            self.recognition_event = Clock.schedule_interval(
                self.perform_recognition, self.recognition_interval
            )
            self.recognition_button.text = "Reconocimiento: ON"
            self.recognition_button.background_color = (0.2, 0.7, 0.3, 1)
            self.add_log("Reconocimiento automático activado")
    
    def perform_recognition(self, dt=None):
        """Realiza el reconocimiento de placas"""
        if self.current_image is None:
            return
        
        # Crear un hilo para el procesamiento para no bloquear la UI
        threading.Thread(target=self.process_image_for_recognition).start()
    
    def process_image_for_recognition(self):
        """Procesa la imagen para reconocer placas (ejecutado en un hilo separado)"""
        try:
            # Hacer una copia de la imagen actual
            img = self.current_image.copy()
            
            # Reducir la resolución para procesamiento más rápido si está activado
            if self.use_low_resolution:
                img = cv2.resize(img, (640, 480))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # MÉTODO OPTIMIZADO PARA PLACAS BOLIVIANAS
            if self.use_fast_detection:
                # Usar un método más rápido y específico para placas bolivianas
                plate_text = self.fast_plate_detection(gray, img)
                if plate_text:
                    # Verificar si es la misma placa que ya reconocimos recientemente
                    current_time = time.time()
                    if (plate_text != self.last_recognized_plate or 
                        current_time - self.last_recognition_time > self.cooldown_period):
                        
                        self.last_recognized_plate = plate_text
                        self.last_recognition_time = current_time
                        
                        # Buscar en la base de datos en el hilo principal
                        Clock.schedule_once(partial(self.search_plate_in_db, plate_text))
            else:
                # Método original más completo pero más lento
                self.original_plate_detection(gray, img)
        
        except Exception as e:
            # Manejar errores en el hilo principal
            Clock.schedule_once(partial(self.handle_recognition_error, str(e)))
    
    def fast_plate_detection(self, gray, img):
        """Método rápido y optimizado para detectar placas bolivianas"""
        # 1. Aplicar umbral adaptativo (más rápido que otros métodos)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Encontrar contornos - usar RETR_EXTERNAL es más rápido
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Filtrar contornos por tamaño y forma
        possible_plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Las placas bolivianas tienen un aspect ratio aproximado entre 2.0 y 4.5
            if 2.0 < aspect_ratio < 4.5 and w > 100 and h > 30:
                possible_plates.append((x, y, w, h))
        
        # 4. Ordenar por área (mayor a menor)
        possible_plates.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        # 5. Procesar las posibles placas (limitado a las 3 más grandes para velocidad)
        for i, (x, y, w, h) in enumerate(possible_plates[:3]):
            # Extraer la región de la placa
            plate_roi = gray[y:y+h, x:x+w]
            
            # Ignorar la parte superior (donde está "BOLIVIA")
            h_roi, w_roi = plate_roi.shape
            plate_roi = plate_roi[int(h_roi*0.3):, :]
            
            # Preprocesamiento rápido
            plate_roi = cv2.resize(plate_roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            _, plate_roi = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR con configuración optimizada para velocidad
            plate_text = pytesseract.image_to_string(
                plate_roi, 
                config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            )
            
            # Limpiar el texto
            plate_text = ''.join(e for e in plate_text if e.isalnum())
            
            # Si encontramos un texto que parece una placa (al menos 6 caracteres)
            if len(plate_text) >= 6:
                # Dibujar el rectángulo en la imagen
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Actualizar la imagen en el hilo principal
                Clock.schedule_once(partial(self.update_recognized_image, img))
                
                return plate_text
        
        # Si no encontramos placas con el método rápido, intentar con patrones
        # Aplicar OCR a toda la imagen con configuración rápida
        full_text = pytesseract.image_to_string(
            gray, 
            config='--psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )
        
        # Buscar patrones de placas bolivianas
        # Patrones: 4 dígitos seguidos de 3 letras o 3 letras seguidas de 4 dígitos
        plate_pattern1 = re.compile(r'\d{4}[A-Z]{3}')
        plate_pattern2 = re.compile(r'[A-Z]{3}\d{4}')
        
        matches1 = plate_pattern1.findall(full_text)
        matches2 = plate_pattern2.findall(full_text)
        
        matches = matches1 + matches2
        
        if matches:
            return matches[0]
        
        return None
    
    def original_plate_detection(self, gray, img):
        """Método original más completo pero más lento"""
        # Aplicar filtro bilateral para reducir ruido pero preservar bordes
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Detectar bordes
        edged = cv2.Canny(gray, 30, 200)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contour = None
        plate_roi = None
        
        # Buscar contornos con 4 vértices (posibles placas)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Si el contorno tiene 4 vértices, podría ser una placa
            if len(approx) == 4:
                plate_contour = approx
                
                # Crear una máscara para extraer solo la placa
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [plate_contour], 0, 255, -1)
                
                # Extraer la región de interés (ROI)
                plate_roi = cv2.bitwise_and(gray, gray, mask=mask)
                
                # Obtener las coordenadas del rectángulo que contiene la placa
                (x, y, w, h) = cv2.boundingRect(plate_contour)
                
                # Extraer solo la región de la placa
                plate_roi = gray[y:y+h, x:x+w]
                break
        
        # Si se encontró un posible contorno de placa
        if plate_roi is not None:
            # Dividir la placa en dos partes: superior (BOLIVIA) e inferior (números/letras)
            h, w = plate_roi.shape
            
            # Ignorar la parte superior (aproximadamente 30% donde está "BOLIVIA")
            # Solo procesar la parte inferior donde están los números y letras
            lower_part = plate_roi[int(h*0.3):h, 0:w]
            
            # Preprocesar la imagen para mejorar el OCR
            lower_part = cv2.resize(lower_part, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, lower_part = cv2.threshold(lower_part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Aplicar OCR solo a la parte inferior
            plate_text = pytesseract.image_to_string(
                lower_part, 
                config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            )
            
            # Limpiar el texto (eliminar espacios, saltos de línea, etc.)
            plate_text = ''.join(e for e in plate_text if e.isalnum())
            
            # Verificar si el texto tiene al menos 5 caracteres (para evitar falsos positivos)
            if len(plate_text) >= 5:
                # Verificar si es la misma placa que ya reconocimos recientemente
                current_time = time.time()
                if (plate_text != self.last_recognized_plate or 
                    current_time - self.last_recognition_time > self.cooldown_period):
                    
                    self.last_recognized_plate = plate_text
                    self.last_recognition_time = current_time
                    
                    # Dibujar el contorno de la placa en la imagen
                    img_with_plate = img.copy()
                    cv2.drawContours(img_with_plate, [plate_contour], -1, (0, 255, 0), 3)
                    cv2.putText(img_with_plate, plate_text, (plate_contour[0][0][0], plate_contour[0][0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Actualizar la imagen en el hilo principal
                    Clock.schedule_once(partial(self.update_recognized_image, img_with_plate))
                    
                    # Buscar en la base de datos en el hilo principal
                    Clock.schedule_once(partial(self.search_plate_in_db, plate_text))
    
    def update_recognized_image(self, img, dt):
        """Actualiza la imagen con la placa reconocida (en el hilo principal)"""
        # Convertir el frame para mostrarlo en Kivy
        buf = cv2.flip(img, 0)  # Voltear verticalmente
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        
        # Actualizar la imagen
        self.camera_image.texture = texture
    
    def search_plate_in_db(self, plate_text, dt=None):
        """Busca la placa en la base de datos (en el hilo principal)"""
        # Limpiar el texto de la placa
        plate_text = plate_text.strip().upper()
        
        self.add_log(f"Buscando placa: {plate_text}")
        
        # Buscar coincidencias parciales
        conn = sqlite3.connect('parking_db.sqlite')
        cursor = conn.cursor()
        
        # Buscar coincidencias exactas primero
        cursor.execute("SELECT * FROM vehicles WHERE plate = ?", (plate_text,))
        vehicle = cursor.fetchone()
        
        # Si no hay coincidencia exacta, buscar coincidencias parciales
        if not vehicle:
            cursor.execute("SELECT * FROM vehicles WHERE plate LIKE ?", (f"%{plate_text}%",))
            vehicle = cursor.fetchone()
        
        conn.close()
        
        if vehicle:
            # Actualizar la interfaz con los datos del vehículo
            self.plate_text = vehicle[1]  # Placa
            self.vehicle_text = vehicle[2]  # Modelo
            self.owner_text = vehicle[3]  # Propietario
            self.phone_text = vehicle[4]  # Teléfono
            self.spot_text = vehicle[5]  # Lugar de estacionamiento
            
            # Actualizar los campos de texto
            self.plate_input.text = self.plate_text
            self.vehicle_input.text = self.vehicle_text
            self.owner_input.text = self.owner_text
            self.phone_input.text = self.phone_text
            self.spot_input.text = self.spot_text
            
            self.add_log(f"Placa reconocida: {vehicle[1]}")
            self.add_log(f"Vehículo: {vehicle[2]}")
            self.add_log(f"Propietario: {vehicle[3]}")
            self.add_log(f"Lugar asignado: {vehicle[5]}")
        else:
            self.add_log(f"Placa {plate_text} no encontrada en la base de datos", "warning")
    
    def handle_recognition_error(self, error_msg, dt):
        """Maneja errores de reconocimiento (en el hilo principal)"""
        self.add_log(f"Error en el reconocimiento: {error_msg}", "error")
    
    def send_whatsapp_direct(self, instance):
        """Envía un mensaje de WhatsApp directamente al propietario del vehículo"""
        phone = self.phone_text
        owner = self.owner_text
        spot = self.spot_text
        
        if not phone or not owner or not spot:
            self.show_error("No hay información completa del vehículo")
            return
        
        try:
            # Mensaje a enviar
            message = f"Hola {owner}, su vehículo ha sido identificado en el estacionamiento. Su lugar asignado es: {spot}"
            
            # Formatear el número de teléfono correctamente para Bolivia
            # Eliminar el "+" si existe
            if phone.startswith('+'):
                phone = phone[1:]
            
            # Asegurarse de que el número tenga el formato correcto para Bolivia
            # Los números de Bolivia comienzan con 591
            if not phone.startswith('591'):
                phone = '591' + phone
            
            # Abrir WhatsApp directamente con URI scheme
            # Este método es más confiable en dispositivos móviles
            whatsapp_url = f"whatsapp://send?phone={phone}&text={urllib.parse.quote(message)}"
            
            self.add_log(f"Abriendo WhatsApp para {phone}")
            webbrowser.open(whatsapp_url)
            
            self.add_log(f"Mensaje preparado para {owner}")
            self.show_info(f"WhatsApp abierto con mensaje para {owner}")
            
        except Exception as e:
            self.add_log(f"Error al abrir WhatsApp: {str(e)}", "error")
            self.show_error(f"Error al abrir WhatsApp: {str(e)}")
    
    def send_whatsapp_web(self, instance):
        """Envía un mensaje de WhatsApp usando WhatsApp Web"""
        phone = self.phone_text
        owner = self.owner_text
        spot = self.spot_text
        
        if not phone or not owner or not spot:
            self.show_error("No hay información completa del vehículo")
            return
        
        try:
            # Mensaje a enviar
            message = f"Hola {owner}, su vehículo ha sido identificado en el estacionamiento. Su lugar asignado es: {spot}"
            
            # Formatear el número de teléfono correctamente para Bolivia
            # Eliminar el "+" si existe
            if phone.startswith('+'):
                phone = phone[1:]
            
            # Asegurarse de que el número tenga el formato correcto para Bolivia
            if not phone.startswith('591'):
                phone = '591' + phone
            
            # Abrir WhatsApp Web
            web_url = f"https://web.whatsapp.com/send?phone={phone}&text={urllib.parse.quote(message)}"
            
            self.add_log(f"Abriendo WhatsApp Web para {phone}")
            webbrowser.open(web_url)
            
            self.add_log(f"Mensaje preparado para {owner} en WhatsApp Web")
            self.show_info(f"WhatsApp Web abierto con mensaje para {owner}")
            
        except Exception as e:
            self.add_log(f"Error al abrir WhatsApp Web: {str(e)}", "error")
            self.show_error(f"Error al abrir WhatsApp Web: {str(e)}")
    
    def add_log(self, message, level="info"):
        """Agrega un mensaje al registro de actividad"""
        log = LogMessage(message, level)
        self.logs.append(log)
        
        # Crear un label para el log con color según el nivel
        color = {
            "info": (1, 1, 1, 1),
            "warning": (1, 0.8, 0, 1),
            "error": (1, 0.3, 0.3, 1)
        }.get(level, (1, 1, 1, 1))
        
        log_label = Label(
            text=str(log),
            color=color,
            size_hint_y=None,
            height=30,
            text_size=(self.width, None),
            halign='left'
        )
        
        self.log_layout.add_widget(log_label)
        
        # Limitar el número de logs mostrados
        if len(self.log_layout.children) > 100:
            self.log_layout.remove_widget(self.log_layout.children[0])
    
    def show_error(self, message):
        """Muestra un popup de error"""
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()
    
    def show_info(self, message):
        """Muestra un popup informativo"""
        popup = Popup(
            title='Información',
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()

if __name__ == '__main__':
    try:
        PlateRecognitionApp().run()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {str(e)}")