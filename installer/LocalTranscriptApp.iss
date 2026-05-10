; Inno Setup script for the Windows installer executable.
; Build after preparing a distribution folder that contains app.py, backend,
; engines, scripts, requirements, and the Python runtime/venv strategy chosen
; for release packaging.

#define MyAppName "Local Transcript App"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Local Transcript App"
#define MyAppExeName "run.bat"

[Setup]
AppId={{7D0E9EF4-5703-4F2D-9C8F-8E0F0A6C4E10}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\LocalTranscriptApp
DefaultGroupName={#MyAppName}
OutputBaseFilename=LocalTranscriptAppSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "venv\*,.git\*,storage\*,models\*,speaker-aware-ai\*,test-transcript-service\*,__pycache__\*,*.pyc"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"
Name: "downloadmodels"; Description: "Download and prepare local models after install"; GroupDescription: "Model setup:"

[Run]
Filename: "{cmd}"; Parameters: "/c python -m venv venv"; WorkingDir: "{app}"; StatusMsg: "Creating Python environment..."; Flags: runhidden waituntilterminated
Filename: "{cmd}"; Parameters: "/c venv\Scripts\python.exe -m pip install --upgrade pip"; WorkingDir: "{app}"; StatusMsg: "Updating installer tools..."; Flags: runhidden waituntilterminated
Filename: "{cmd}"; Parameters: "/c venv\Scripts\pip.exe install openvino==2026.1.0"; WorkingDir: "{app}"; StatusMsg: "Installing OpenVINO..."; Flags: waituntilterminated
Filename: "{cmd}"; Parameters: "/c venv\Scripts\pip.exe install -r requirements.txt"; WorkingDir: "{app}"; StatusMsg: "Installing app dependencies..."; Flags: waituntilterminated
Filename: "{cmd}"; Parameters: "/c venv\Scripts\pip.exe uninstall torchcodec -y"; WorkingDir: "{app}"; StatusMsg: "Finalizing dependencies..."; Flags: runhidden waituntilterminated
Filename: "{cmd}"; Parameters: "/c venv\Scripts\python.exe scripts\bootstrap_models.py"; WorkingDir: "{app}"; StatusMsg: "Checking hardware and downloading local models..."; Flags: waituntilterminated; Tasks: downloadmodels
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: postinstall skipifsilent nowait
