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
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

class MusicTranscriber:
    def __init__(self, min_freq=80, max_freq=2000, threshold=0.1, sensitivity=0.5):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.tempo = None
        
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
    
    def detect_tempo(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
            
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            
            tempo = int(round(float(tempo)))
            tempo = max(40, min(240, tempo))
            
            self.tempo = tempo
            return tempo
            
        except Exception as e:
            print(f"Ошибка определения темпа: {e}")
            return 120
    
    def detect_notes_with_frequencies(self, audio_file, progress_callback=None):
        y, sr = librosa.load(audio_file)
        
        if progress_callback:
            progress_callback(10, "Загрузка аудио...")
        
        D = librosa.stft(y)
        frequencies = librosa.fft_frequencies(sr=sr)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        notes = []
        times = []
        velocities = []
        
        hop_length = 512
        total_frames = D.shape[1]
        
        threshold_db = -60 + (self.sensitivity * 40)
        
        if progress_callback:
            progress_callback(20, "Анализ частот...")
        
        for i in range(total_frames):
            magnitude_db = D_db[:, i]
            max_idx = np.argmax(magnitude_db)
            freq = frequencies[max_idx]
            amplitude_db = magnitude_db[max_idx]
            
            if (freq > self.min_freq and 
                freq < self.max_freq and 
                amplitude_db > threshold_db):
                
                midi_note = self.freq_to_midi(freq)
                if midi_note is not None:
                    notes.append(int(round(midi_note)))
                    amplitude_linear = librosa.db_to_amplitude(amplitude_db)
                    velocity = int(min(127, max(1, amplitude_linear * 127 * (1.5 - self.sensitivity * 0.5))))
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
                progress = 30 + (i / total_frames) * 50
                progress_callback(progress, f"Обработка кадра {i}/{total_frames}")
        
        if progress_callback:
            progress_callback(80, "Анализ завершен")
        
        return notes, times, velocities
    
    def group_notes_with_duration(self, notes, times, velocities, min_duration=0.05, max_gap=0.1):
        grouped = []
        
        if not notes:
            return grouped
        
        current_note = notes[0]
        current_velocity = velocities[0]
        start_time = times[0]
        last_time = times[0]
        last_velocity = velocities[0]
        
        for i in range(1, len(notes)):
            time_gap = times[i] - last_time
            
            if notes[i] != current_note or time_gap > max_gap:
                duration = last_time - start_time
                
                if duration >= min_duration and current_note is not None:
                    avg_velocity = int((current_velocity + last_velocity) / 2)
                    
                    grouped.append({
                        'note': current_note,
                        'start': start_time,
                        'duration': duration,
                        'velocity': avg_velocity,
                        'end': last_time
                    })
                
                current_note = notes[i]
                current_velocity = velocities[i]
                start_time = times[i]
            
            last_time = times[i]
            last_velocity = velocities[i]
        
        if current_note is not None:
            duration = last_time - start_time
            if duration >= min_duration:
                avg_velocity = int((current_velocity + last_velocity) / 2)
                grouped.append({
                    'note': current_note,
                    'start': start_time,
                    'duration': duration,
                    'velocity': avg_velocity,
                    'end': last_time
                })
        
        return grouped
    
    def convert_duration_to_beats(self, duration_seconds, tempo):
        if tempo <= 0:
            return 0
        seconds_per_beat = 60 / tempo
        beats = duration_seconds / seconds_per_beat
        return beats
    
    def get_note_type(self, duration_beats):
        if duration_beats >= 4:
            return "Целая"
        elif duration_beats >= 3:
            return "Половинная с точкой"
        elif duration_beats >= 2:
            return "Половинная"
        elif duration_beats >= 1.5:
            return "Четвертная с точкой"
        elif duration_beats >= 1:
            return "Четвертная"
        elif duration_beats >= 0.75:
            return "Восьмая с точкой"
        elif duration_beats >= 0.5:
            return "Восьмая"
        elif duration_beats >= 0.375:
            return "Шестнадцатая с точкой"
        elif duration_beats >= 0.25:
            return "Шестнадцатая"
        else:
            return "Тридцать вторая"
    
    def analyze_durations(self, grouped_notes):
        if not grouped_notes:
            return
        
        durations = [note['duration'] for note in grouped_notes]
        
        stats = {
            'total_notes': len(grouped_notes),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'median_duration': np.median(durations),
            'short_notes': sum(1 for d in durations if d < 0.2),
            'medium_notes': sum(1 for d in durations if 0.2 <= d < 0.5),
            'long_notes': sum(1 for d in durations if d >= 0.5)
        }
        
        return stats
    
    def save_to_midi(self, grouped_notes, output_file, tempo=None):
        if tempo is None:
            tempo = self.tempo if self.tempo else 120
        
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
        self.root.geometry("1100x850")
        
        self.transcriber = MusicTranscriber()
        self.audio_file = None
        self.grouped_notes = None
        
        self.setup_ui()
        
    def setup_ui(self):
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
        
        # Темп
        tempo_frame = ttk.Frame(settings_frame)
        tempo_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(tempo_frame, text="Темп (BPM):").grid(row=0, column=0, sticky=tk.W)
        self.tempo_var = tk.IntVar(value=120)
        self.tempo_spinbox = ttk.Spinbox(tempo_frame, from_=40, to=240, textvariable=self.tempo_var, width=10)
        self.tempo_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        self.auto_tempo_btn = ttk.Button(tempo_frame, text="🎵 Автоопределение темпа", 
                                        command=self.auto_detect_tempo)
        self.auto_tempo_btn.grid(row=0, column=2, padx=5)
        
        self.tempo_status_var = tk.StringVar(value="")
        ttk.Label(tempo_frame, textvariable=self.tempo_status_var, foreground="green").grid(row=0, column=3, padx=5)
        
        # Чувствительность
        sensitivity_frame = ttk.Frame(settings_frame)
        sensitivity_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(sensitivity_frame, text="Чувствительность:").grid(row=0, column=0, sticky=tk.W)
        
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        self.sensitivity_scale = ttk.Scale(
            sensitivity_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.sensitivity_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_sensitivity_label
        )
        self.sensitivity_scale.grid(row=0, column=1, padx=5)
        
        self.sensitivity_label = ttk.Label(sensitivity_frame, text="Средняя")
        self.sensitivity_label.grid(row=0, column=2, padx=5)
        
        ttk.Label(sensitivity_frame, text="(0 - очень чувствительно, 1 - менее чувствительно)").grid(row=0, column=3, padx=5)
        
        # Частотный диапазон
        freq_frame = ttk.Frame(settings_frame)
        freq_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(freq_frame, text="Мин. частота (Гц):").grid(row=0, column=0, sticky=tk.W)
        self.min_freq_var = tk.IntVar(value=80)
        ttk.Spinbox(freq_frame, from_=20, to=500, textvariable=self.min_freq_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(freq_frame, text="Макс. частота (Гц):").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.max_freq_var = tk.IntVar(value=2000)
        ttk.Spinbox(freq_frame, from_=500, to=10000, textvariable=self.max_freq_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Кнопки
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.transcribe_btn = ttk.Button(buttons_frame, text="🎵 Транскрибировать", command=self.start_transcription)
        self.transcribe_btn.grid(row=0, column=0, padx=5)
        
        self.save_btn = ttk.Button(buttons_frame, text="💾 Сохранить MIDI", command=self.save_midi, state='disabled')
        self.save_btn.grid(row=0, column=1, padx=5)
        
        self.play_btn = ttk.Button(buttons_frame, text="▶️ Воспроизвести оригинал", command=self.play_audio, state='disabled')
        self.play_btn.grid(row=0, column=2, padx=5)
        
        # Прогресс
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        # Результаты
        results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=100, height=20)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # График
        self.figure = plt.Figure(figsize=(10, 4), dpi=100)
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
        
    def update_sensitivity_label(self, value):
        sensitivity = float(value)
        if sensitivity < 0.2:
            label = "Очень чувствительно"
        elif sensitivity < 0.4:
            label = "Чувствительно"
        elif sensitivity < 0.6:
            label = "Средняя"
        elif sensitivity < 0.8:
            label = "Низкая"
        else:
            label = "Очень низкая"
        
        self.sensitivity_label.config(text=label)
        
    def auto_detect_tempo(self):
        if not self.audio_file:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите аудиофайл")
            return
        
        self.auto_tempo_btn.config(state='disabled')
        self.tempo_status_var.set("Определение...")
        self.root.update()
        
        try:
            thread = threading.Thread(target=self._detect_tempo_thread)
            thread.start()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось определить темп: {str(e)}")
            self.auto_tempo_btn.config(state='normal')
            self.tempo_status_var.set("")
    
    def _detect_tempo_thread(self):
        try:
            tempo = self.transcriber.detect_tempo(self.audio_file)
            self.tempo_var.set(tempo)
            self.tempo_status_var.set(f"Определен: {tempo} BPM")
            self.root.after(3000, lambda: self.tempo_status_var.set(""))
        except Exception as e:
            self.tempo_status_var.set("Ошибка определения")
            messagebox.showerror("Ошибка", f"Не удалось определить темп: {str(e)}")
        finally:
            self.auto_tempo_btn.config(state='normal')
        
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
        
        self.transcriber.min_freq = self.min_freq_var.get()
        self.transcriber.max_freq = self.max_freq_var.get()
        self.transcriber.sensitivity = self.sensitivity_var.get()
        
        self.transcribe_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        thread = threading.Thread(target=self.transcribe_audio)
        thread.start()
        
    def transcribe_audio(self):
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Начинаю транскрипцию...\n")
            
            notes, times, velocities = self.transcriber.detect_notes_with_frequencies(
                self.audio_file, 
                progress_callback=self.update_progress
            )
            
            self.update_progress(85, "Группировка нот...")
            self.grouped_notes = self.transcriber.group_notes_with_duration(notes, times, velocities)
            
            self.update_progress(95, "Сохранение результатов...")
            
            # Статистика длительности
            stats = self.transcriber.analyze_durations(self.grouped_notes)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== РЕЗУЛЬТАТЫ ТРАНСКРИПЦИИ ===\n\n")
            self.results_text.insert(tk.END, f"Распознано нот: {stats['total_notes']}\n")
            self.results_text.insert(tk.END, f"Темп: {self.tempo_var.get()} BPM\n")
            self.results_text.insert(tk.END, f"Чувствительность: {self.sensitivity_var.get():.2f}\n\n")
            
            self.results_text.insert(tk.END, "=== СТАТИСТИКА ДЛИТЕЛЬНОСТИ ===\n")
            self.results_text.insert(tk.END, f"Средняя длительность: {stats['avg_duration']:.3f} сек\n")
            self.results_text.insert(tk.END, f"Минимальная: {stats['min_duration']:.3f} сек\n")
            self.results_text.insert(tk.END, f"Максимальная: {stats['max_duration']:.3f} сек\n")
            self.results_text.insert(tk.END, f"Короткие (<0.2с): {stats['short_notes']}\n")
            self.results_text.insert(tk.END, f"Средние (0.2-0.5с): {stats['medium_notes']}\n")
            self.results_text.insert(tk.END, f"Длинные (>0.5с): {stats['long_notes']}\n\n")
            
            self.results_text.insert(tk.END, "=== ПЕРВЫЕ 50 НОТ ===\n")
            self.results_text.insert(tk.END, "-" * 90 + "\n")
            self.results_text.insert(tk.END, f"{'№':<4} {'Нота':<8} {'Начало(с)':<12} {'Длит.(с)':<12} {'Длит.(доли)':<12} {'Тип':<15} {'Vel':<5}\n")
            self.results_text.insert(tk.END, "-" * 90 + "\n")
            
            tempo = self.tempo_var.get()
            
            for i, note_data in enumerate(self.grouped_notes[:50]):
                note_name = self.transcriber.midi_to_note(note_data['note'])
                duration_beats = self.transcriber.convert_duration_to_beats(note_data['duration'], tempo)
                note_type = self.transcriber.get_note_type(duration_beats)
                
                line = f"{i+1:<4} {note_name:<8} {note_data['start']:<12.2f} {note_data['duration']:<12.3f} {duration_beats:<12.2f} {note_type:<15} {note_data['velocity']:<5}\n"
                self.results_text.insert(tk.END, line)
            
            self.display_plot()
            
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
            for note_data in self.grouped_notes:
                start = note_data['start']
                duration = note_data['duration']
                
                # Цвет зависит от длительности
                if duration < 0.2:
                    color = 'red'
                elif duration < 0.5:
                    color = 'blue'
                else:
                    color = 'green'
                
                # Рисуем прямоугольник для ноты
                self.ax.barh(y=note_data['note'], 
                           width=duration, 
                           left=start, 
                           height=0.8,
                           color=color, 
                           alpha=0.7)
            
            self.ax.set_xlabel('Время (секунды)')
            self.ax.set_ylabel('MIDI нота')
            self.ax.set_title('Длительность нот (красный - короткие, синий - средние, зеленый - длинные)')
            self.ax.grid(True, alpha=0.3)
            
            # Легенда
            legend_elements = [
                Patch(facecolor='red', label='Короткие (< 0.2с)'),
                Patch(facecolor='blue', label='Средние (0.2-0.5с)'),
                Patch(facecolor='green', label='Длинные (> 0.5с)')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right')
            
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
                if platform.system() == 'Darwin':
                    os.system(f'open "{self.audio_file}"')
                elif platform.system() == 'Windows':
                    os.system(f'start "" "{self.audio_file}"')
                else:
                    os.system(f'xdg-open "{self.audio_file}"')
            except:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести файл")

def main():
    root = tk.Tk()
    app = MusicTranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
