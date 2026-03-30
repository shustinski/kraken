; Build with:
;   ISCC NeuralImageInstaller.iss
; or override defaults:
;   ISCC /DAppVersion=5.3.0 /DBuildDir="dist\NeuralImage 5.4" NeuralImageInstaller.iss

#ifndef AppName
  #define AppName "NeuralImage"
#endif

#ifndef AppVersion
  #define AppVersion "5.9.1"
#endif

#ifndef AppPublisher
  #define AppPublisher "NeuralImage"
#endif

#ifndef AppExeName
  #define AppExeName "NeuralImage.exe"
#endif

#ifndef BuildDir
  #define BuildDir "dist\\NeuralImage"
#endif

#ifndef OutputDir
  #define OutputDir "dist\\installer"
#endif

[Setup]
AppId={{9D56A9E1-465A-4C2A-9FBA-ED7E0060F3C0}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf64}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
CompressionThreads=auto
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename={#AppName}-{#AppVersion}
SetupIconFile={#BuildDir}\_internal\icon.ico
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
PrivilegesRequired=admin
DisableProgramGroupPage=yes
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
