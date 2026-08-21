import os
import sys
import numpy as np
import librosa
import mido
from mido import MidiFile, MidiTrack, Message
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class MusicTranscriber:
    def __init__(self, min_freq=80, max_freq=2000, threshold=0.1):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.threshold = threshold
        
    def freq_to_midi(self, freq):
        if freq <= 0:
            return None
        return 69 + 12 * np.log2(freq / 440.0)
    
    def midi_to_note(self, midi):
        if midi is None:
            return "Rest"
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int(midi // 12) - 1
        note = note_names[int(midi % 12)]
        return f"{note}{octave}"
    
    def detect_notes_with_frequencies(self, audio_file, progress_callback=None):
        y, sr = librosa.load(audio_file)
        
        if progress_callback:
            progress_callback(20, "Анализ аудио...")
        
        D = librosa.stft(y)
        frequencies = librosa.fft_frequencies(sr=sr)
        
        notes = []
        times = []
        velocities = []
        
        hop_length = 512
        
        total_frames = D.shape[1]
        
        for i in range(total_frames):
            magnitude = np.abs(D[:, i])
            max_idx = np.argmax(magnitude)
            freq = frequencies[max_idx]
            amplitude = magnitude[max_idx]
            
            if freq > self.min_freq and freq < self.max_freq and amplitude > self.threshold:
                midi_note = self.freq_to_midi(freq)
                if midi_note is not None:
                    notes.append(int(round(midi_note)))
                    velocity = int(min(127, max(1, amplitude * 127)))
                    velocities.append(velocity)
                else:
                    notes.append(None)
                    velocities.append(0)
            else:
                notes.append(None)
                velocities.append(0)
            
            time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            times.append(time)
            
            if progress_callback and i % 1000 == 0:
                progress = 20 + (i / total_frames) * 60
                progress_callback(progress, f"Обработка кадра {i}/{total_frames}")
        
        if progress_callback:
            progress_callback(80, "Анализ завершен")
        
        return notes, times, velocities
    
    def group_notes_with_duration(self, notes, times, velocities, min_duration=0.05):
        grouped = []
        
        if not notes:
            return grouped
        
        current_note = notes[0]
        current_velocity = velocities[0]
        start_time = times[0]
        last_time = times[0]
        
        for i in range(1, len(notes)):
            if notes[i] != current_note:
                duration = last_time - start_time
                if duration >= min_duration and current_note is not None:
                    grouped.append({
                        'note': current_note,
                        'start': start_time,
                        'duration': duration,
                        'velocity': current_velocity
                    })
                current_note = notes[i]
                current_velocity = velocities[i]
                start_time = times[i]
            
            last_time = times[i]
        
        if current_note is not None:
            duration = last_time - start_time
            if duration >= min_duration:
                grouped.append({
                    'note': current_note,
                    'start': start_time,
                    'duration': duration,
                    'velocity': current_velocity
                })
        
        return grouped
    
    def save_to_midi(self, grouped_notes, output_file, tempo=120):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        tempo_microseconds = int(60_000_000 / tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))
        track.append(Message('program_change', program=0, time=0))
        
        ticks_per_beat = 480
        seconds_per_beat = 60 / tempo
        ticks_per_second = ticks_per_beat / seconds_per_beat
        
        grouped_notes.sort(key=lambda x: x['start'])
        
        current_time_ticks = 0
        
        for note_data in grouped_notes:
            start_ticks = int(note_data['start'] * ticks_per_second)
            duration_ticks = int(note_data['duration'] * ticks_per_second)
            
            delta_time = start_ticks - current_time_ticks
            
            track.append(Message('note_on', 
                               note=note_data['note'], 
                               velocity=note_data['velocity'], 
                               time=delta_time))
            
            track.append(Message('note_off', 
                               note=note_data['note'], 
                               velocity=0, 
                               time=duration_ticks))
            
            current_time_ticks = start_ticks + duration_ticks
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        mid.save(output_file)
        
        return output_file

class MusicTranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Музыкальный транскрайбер")
        self.root.geometry("900x700")
        
        self.transcriber = MusicTranscriber()
        self.audio_file = None
        self.grouped_notes = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройки
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding="10")
        settings_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Выбор файла
        ttk.Label(settings_frame, text="Аудиофайл:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(settings_frame, text="Обзор...", command=self.browse_file).grid(row=0, column=2)
        
        # Параметры
        ttk.Label(settings_frame, text="Темп (BPM):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.tempo_var = tk.IntVar(value=120)
        ttk.Spinbox(settings_frame, from_=40, to=240, textvariable=self.tempo_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(settings_frame, text="Мин. частота (Гц):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.min_freq_var = tk.IntVar(value=80)
        ttk.Spinbox(settings_frame, from_=20, to=500, textvariable=self.min_freq_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(settings_frame, text="Макс. частота (Гц):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.max_freq_var = tk.IntVar(value=2000)
        ttk.Spinbox(settings_frame, from_=500, to=10000, textvariable=self.max_freq_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Кнопки
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.transcribe_btn = ttk.Button(buttons_frame, text="🎵 Транскрибировать", command=self.start_transcription)
        self.transcribe_btn.grid(row=0, column=0, padx=5)
        
        self.save_btn = ttk.Button(buttons_frame, text="💾 Сохранить MIDI", command=self.save_midi, state='disabled')
        self.save_btn.grid(row=0, column=1, padx=5)
        
        self.play_btn = ttk.Button(buttons_frame, text="▶️ Воспроизвести оригинал", command=self.play_audio, state='disabled')
        self.play_btn.grid(row=0, column=2, padx=5)
        
        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        # Результаты
        results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Текстовое поле для результатов
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=15)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # График
        self.figure = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, results_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Настройка grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудио файлы", "*.mp3 *.wav *.flac *.m4a *.ogg"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            self.audio_file = filename
            self.file_path_var.set(filename)
            self.play_btn.config(state='normal')
            
    def update_progress(self, value, status):
        self.progress_var.set(value)
        self.status_var.set(status)
        self.root.update_idletasks()
        
    def start_transcription(self):
        if not self.audio_file:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите аудиофайл")
            return
        
        # Обновляем параметры транскрайбера
        self.transcriber.min_freq = self.min_freq_var.get()
        self.transcriber.max_freq = self.max_freq_var.get()
        
        # Отключаем кнопки
        self.transcribe_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=self.transcribe_audio)
        thread.start()
        
    def transcribe_audio(self):
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Начинаю транскрипцию...\n")
            
            # Транскрипция
            notes, times, velocities = self.transcriber.detect_notes_with_frequencies(
                self.audio_file, 
                progress_callback=self.update_progress
            )
            
            self.update_progress(85, "Группировка нот...")
            self.grouped_notes = self.transcriber.group_notes_with_duration(notes, times, velocities)
            
            self.update_progress(95, "Сохранение результатов...")
            
            # Вывод результатов
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Распознано нот: {len(self.grouped_notes)}\n\n")
            self.results_text.insert(tk.END, "Первые 50 нот:\n")
            self.results_text.insert(tk.END, "-" * 60 + "\n")
            
            for i, note_data in enumerate(self.grouped_notes[:50]):
                note_name = self.transcriber.midi_to_note(note_data['note'])
                line = f"{i+1:3d}. {note_name:5s} | Начало: {note_data['start']:6.2f}s | Длительность: {note_data['duration']:5.2f}s | Velocity: {note_data['velocity']:3d}\n"
                self.results_text.insert(tk.END, line)
            
            # Отображение графика
            self.display_plot()
            
            # Включаем кнопки
            self.save_btn.config(state='normal')
            self.update_progress(100, "Транскрипция завершена!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.update_progress(0, "Ошибка")
        finally:
            self.transcribe_btn.config(state='normal')
            
    def display_plot(self):
        self.ax.clear()
        
        if self.grouped_notes:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            for note_data in self.grouped_notes:
                note = note_data['note'] % 12
                start = note_data['start']
                duration = note_data['duration']
                
                self.ax.hlines(y=note_data['note'], xmin=start, xmax=start+duration, 
                              colors='blue', linewidth=2)
            
            self.ax.set_xlabel('Время (секунды)')
            self.ax.set_ylabel('MIDI нота')
            self.ax.set_title('Распознанные ноты')
            self.ax.grid(True, alpha=0.3)
            
        self.canvas.draw()
        
    def save_midi(self):
        if not self.grouped_notes:
            messagebox.showerror("Ошибка", "Нет данных для сохранения")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Сохранить MIDI файл",
            defaultextension=".mid",
            filetypes=[("MIDI файлы", "*.mid"), ("Все файлы", "*.*")]
        )
        
        if filename:
            try:
                tempo = self.tempo_var.get()
                self.transcriber.save_to_midi(self.grouped_notes, filename, tempo)
                messagebox.showinfo("Успех", f"MIDI файл сохранен: {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
                
    def play_audio(self):
        if self.audio_file:
            try:
                import platform
                if platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{self.audio_file}"')
                elif platform.system() == 'Windows':  # Windows
                    os.system(f'start "" "{self.audio_file}"')
                else:  # Linux
                    os.system(f'xdg-open "{self.audio_file}"')
            except:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести файл")

def main():
    root = tk.Tk()
    app = MusicTranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()