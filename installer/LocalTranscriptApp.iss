; Inno Setup script for the Windows GUI installer.
;
; Build flow:
;   1. Build the standalone GUI executable:
;        python -m PyInstaller --noconfirm --clean LocalTranscriptApp.spec
;   2. Pre-cache gated models into .\models\ on the BUILD machine using a
;      valid HF_TOKEN when needed, then materialize the HF cache for Inno.
;   3. Compile this script with Inno Setup Compiler (iscc).
;
; The wizard collects optional HF_TOKEN + resource settings and writes them
; into {app}\.env after files are installed.

#define MyAppName "Local Transcript App"
#define MyAppVersion "1.2.7"
#define MyAppPublisher "Local Transcript App"
#define MyAppExeName "LocalTranscriptApp.exe"
#define ModelStageRoot "C:\lta-installer-stage-real"

[Setup]
AppId={{7D0E9EF4-5703-4F2D-9C8F-8E0F0A6C4E10}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\LocalTranscriptApp
DefaultGroupName={#MyAppName}
OutputDir=..\release\v1.2.7
OutputBaseFilename=LocalTranscriptAppSetup
Compression=none
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=1900000000
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\LocalTranscriptApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\.env.production"; DestDir: "{app}"; DestName: ".env"; Flags: onlyifdoesntexist
Source: "{#ModelStageRoot}\models\hf_cache\*"; DestDir: "{app}\models\hf_cache"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ModelStageRoot}\models\torch\*"; DestDir: "{app}\models\torch"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\RELEASE_NOTES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\installer\write_runtime_env.py"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: postinstall skipifsilent nowait

[Code]
var
  TokenPage: TInputQueryWizardPage;
  ResourcePage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  TokenPage := CreateInputQueryPage(wpSelectTasks,
    'Hugging Face token',
    'Optional when models are already bundled under models\hf_cache.',
    'Enter a Hugging Face token only if you need to download gated models (Typhoon / pyannote). Leave blank for offline packs.');
  TokenPage.Add('HF_TOKEN:', False);

  ResourcePage := CreateInputQueryPage(TokenPage.ID,
    'Resource settings',
    'Minimum host: 4 CPU threads / 8 GB RAM. NVIDIA CUDA still needs ≥ 8 GB VRAM.',
    'Tune CPU threads and optional backend force. Empty backend = auto-detect.');
  ResourcePage.Add('APP_CPU_THREADS (0 = auto):', False);
  ResourcePage.Add('APP_FORCE_BACKEND (cuda|rocm|openvino|directml|cpu):', False);
  ResourcePage.Add('OV_DEVICE (GPU|NPU|CPU):', False);
  ResourcePage.Add('MIN_SYSTEM_RAM_MB:', False);
  ResourcePage.Add('MIN_CPU_THREADS:', False);
  ResourcePage.Values[0] := '0';
  ResourcePage.Values[2] := 'GPU';
  ResourcePage.Values[3] := '8192';
  ResourcePage.Values[4] := '4';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Cmd, Token, Threads, Backend, OvDevice, MinRam, MinCpu: string;
begin
  if CurStep <> ssPostInstall then
    Exit;

  Token := Trim(TokenPage.Values[0]);
  Threads := Trim(ResourcePage.Values[0]);
  Backend := Trim(ResourcePage.Values[1]);
  OvDevice := Trim(ResourcePage.Values[2]);
  MinRam := Trim(ResourcePage.Values[3]);
  MinCpu := Trim(ResourcePage.Values[4]);
  if Threads = '' then Threads := '0';
  if MinRam = '' then MinRam := '8192';
  if MinCpu = '' then MinCpu := '4';

  Cmd := '"' + ExpandConstant('{app}\{#MyAppExeName}') + '"';
  { Prefer bundled Python helper via the app folder — fall back to rewriting .env in Pascal if python missing. }
  if FileExists(ExpandConstant('{app}\installer\write_runtime_env.py')) then
  begin
    if Exec('python',
      '"' + ExpandConstant('{app}\installer\write_runtime_env.py') + '"' +
      ' --env-path "' + ExpandConstant('{app}\.env') + '"' +
      ' --hf-token "' + Token + '"' +
      ' --cpu-threads "' + Threads + '"' +
      ' --force-backend "' + Backend + '"' +
      ' --ov-device "' + OvDevice + '"' +
      ' --min-ram-mb "' + MinRam + '"' +
      ' --min-cpu-threads "' + MinCpu + '"' +
      ' --min-vram-mb "8192"' +
      ' --ui-max-jobs "1"' +
      ' --ui-gradio-concurrency "4"',
      ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      Log(Format('write_runtime_env.py exit=%d', [ResultCode]));
    end
    else
      Log('python not found; .env left as shipped template — edit manually.');
  end;
end;
