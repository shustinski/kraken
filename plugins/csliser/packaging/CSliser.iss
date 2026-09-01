#define MyAppName "CSliser"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppPublisher "ViaLaNet"
#define MyAppExeName "CSliser.exe"
#define MyAppDistDir "..\\dist\\CSliser"
#define MyAppIcon "..\\resources\\icons\\csliser.ico"

[Setup]
AppId={{C4E8F1AC-92D7-41B3-A6E1-7088D5C31290}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\ViaLaNet\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir=.
OutputBaseFilename=CSliser-setup-{#MyAppVersion}
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKLM; Subkey: "Software\Kraken\Plugins\csliser"; ValueType: string; ValueName: "Executable"; ValueData: "{app}\{#MyAppExeName}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Kraken\Plugins\csliser"; ValueType: string; ValueName: "DisplayName"; ValueData: "{#MyAppName}"
Root: HKLM; Subkey: "Software\Kraken\Plugins\csliser"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"
Root: HKLM; Subkey: "Software\Kraken\Plugins\csliser"; ValueType: string; ValueName: "ProtocolVersion"; ValueData: "1.0"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
