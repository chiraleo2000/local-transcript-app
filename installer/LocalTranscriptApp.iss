; Inno Setup script for the Windows GUI installer.
;
; Build flow:
;   1. Build the standalone GUI executable:
;        python -m PyInstaller --noconfirm --clean LocalTranscriptApp.spec
;      Output: dist\LocalTranscriptApp\LocalTranscriptApp.exe plus _internal\
;   2. Pre-cache gated models into .\models\ on the BUILD machine using a
;      valid HF_TOKEN, then strip the token from .env.production. The
;      installer copies the resulting cache as part of the payload so end
;      users never need a Hugging Face token.
;   3. Compile this script with Inno Setup Compiler (iscc) to produce
;      LocalTranscriptAppSetup.exe.

#define MyAppName "Local Transcript App"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "Local Transcript App"
#define MyAppExeName "LocalTranscriptApp.exe"

[Setup]
AppId={{7D0E9EF4-5703-4F2D-9C8F-8E0F0A6C4E10}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\LocalTranscriptApp
DefaultGroupName={#MyAppName}
OutputDir=..\release\v1.2.0
OutputBaseFilename=LocalTranscriptAppSetup
Compression=lzma2/max
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=1900000000
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
; GUI application produced by PyInstaller onedir mode.
Source: "..\dist\LocalTranscriptApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Production env template (no HF_TOKEN; offline cache-first).
Source: "..\.env.production"; DestDir: "{app}"; DestName: ".env"; Flags: onlyifdoesntexist
; Pre-cached HF model payload (gated models bundled at build time — no token needed at runtime).
; Do not ship models\_archive or generated OpenVINO caches; they can be huge and stale.
Source: "..\models\hf_cache\*"; DestDir: "{app}\models\hf_cache"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\models\torch\*"; DestDir: "{app}\models\torch"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
; Optional: ship source for advanced users / offline diagnostics.
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\RUN_INSTRUCTIONS.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: postinstall skipifsilent nowait
