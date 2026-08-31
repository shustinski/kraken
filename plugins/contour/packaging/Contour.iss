; Build with:
;   ISCC packaging\Contour.iss
; or from the plugin root:
;   .\scripts\build_windows.ps1
; Override defaults:
;   ISCC /DMyAppVersion=0.9.6 packaging\Contour.iss

; Resolve paths from this script location so ISCC does not depend on cwd.
#define ProjectDir AddBackslash(SourcePath) + ".."

#ifndef MyAppName
  #define MyAppName "Contour"
#endif

#ifndef MyAppVersion
  #define MyAppVersion "0.9.5"
#endif

#ifndef MyAppPublisher
  #define MyAppPublisher "Contour"
#endif

#ifndef MyAppExeName
  #define MyAppExeName "Contour.exe"
#endif

#ifndef MyAppDistDir
  #define MyAppDistDir ProjectDir + "\dist\Contour"
#endif

#ifndef MyAppOutputDir
  #define MyAppOutputDir ProjectDir + "\dist\installer"
#endif

#ifndef MyAppIcon
  #define MyAppIcon ProjectDir + "\resources\icons\contour.ico"
#endif

[Setup]
AppId={{3D57BB57-5DD3-40F8-8521-0FC09E6EF8B5}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Contour\Polygon Widget
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
CompressionThreads=auto
WizardStyle=modern
LanguageDetectionMethod=none
OutputDir={#MyAppOutputDir}
OutputBaseFilename=Contour-setup-{#MyAppVersion}
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoProductName={#MyAppName}
PrivilegesRequired=admin
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
