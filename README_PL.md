# Odtworzenie eksperymentów

Proces uczenia sieci neuronowych zrealizowano w osobnym repozytorium projektu, z wykorzystaniem frameworka Kedro.

Do odtworzenia eksperymentów wymagane jest środowisko **Python 3.12** oraz karta graficzna z obsługą **CUDA**.

Dane medyczne wymagają dużej ilości wolnego miejsca na dysku — około **100 GB**.

## Przygotowanie środowiska

1. Pobierz kod źródłowy:

   ```bash
   git clone https://github.com/psiku/dnn-cxr-diagnostics.git
   ```

2. Pobierz wymagane dane:
   - katalog `images`,
   - `Data_Entry_2017_v2020.csv`,
   - `BBox_List_2017.csv`,
   - `train_val_list.txt`,
   - `test_list.txt`.

   Dane są dostępne pod adresem:

   https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

   Umieść je w katalogu:

   ```text
   data/01_raw
   ```

   W plikach CSV nazwy kolumn należy sprowadzić do konwencji `snake_case`, np.:

   ```text
   Image Index -> image_index
   ```

   Pliki list podziału na zbiory treningowy i testowy nie zawierają nagłówków i nie wymagają tej zmiany.

3. Pobierz dane masek segmentacji CheXmask (`Preprocessed/CheXpert.csv`) ze strony:

   https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/

   Następnie umieść plik w katalogu:

   ```text
   data/01_raw
   ```

## Instalacja środowiska Python

W katalogu głównym repozytorium utwórz wirtualne środowisko Pythona i zainstaluj zależności:

1. Utwórz środowisko:

   ```bash
   python -m venv .venv
   ```

2. Aktywuj środowisko.

   **Windows:**

   ```powershell
   .venv\Scripts\activate
   ```

   **Linux / macOS:**

   ```bash
   source .venv/bin/activate
   ```

3. Zainstaluj zależności:

   ```bash
   pip install -r requirements.txt
   ```

## Konfiguracja i uruchomienie

Parametry eksperymentu określa się w plikach konfiguracyjnych Kedro.

Przykładowa konfiguracja znajduje się w repozytorium, a opis pól — w pliku `README.md`.

### Pełny przebieg potoku

```bash
kedro run --params compute_tensors=true
```

Flaga `compute_tensors` powoduje jednorazowe wyliczenie tensorów `.pt` ze zdjęć. W kolejnych uruchomieniach można ją wyłączyć i korzystać z wcześniej zapisanych plików.

### Wizualizacja potoku

```bash
kedro viz
```

Polecenie uruchamia wizualizację połączeń między węzłami potoku.

### Interfejs MLflow

```bash
kedro mlflow ui
```

Polecenie uruchamia lokalny interfejs MLflow z zapisami przebiegów treningu.
