name: Build SoundToMidi with Tempo Detection

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy==1.24.3
        pip install scipy==1.10.1
        pip install matplotlib==3.7.2
        pip install librosa==0.10.1
        pip install mido==1.3.0
        pip install soundfile==0.12.1
        pip install audioread==3.0.0
        pip install pyinstaller
    
    - name: Build EXE
      shell: pwsh
      run: |
        # Автоматически находим Python файл
        $pyFile = Get-ChildItem -Filter "*.py" | Where-Object { 
          $_.Name -ne "setup.py" -and $_.Name -notlike "test*" 
        } | Select-Object -First 1
        
        if ($pyFile) {
          Write-Host "Building: $($pyFile.Name)"
          pyinstaller --onefile --windowed --clean --name "SoundToMidi" $pyFile.Name
        } else {
          Write-Error "Python file not found"
          exit 1
        }
    
    - name: Upload EXE
      uses: actions/upload-artifact@v4
      with:
        name: SoundToMidi-Windows
        path: dist/SoundToMidi.exe
        if-no-files-found: error
