#define AppVersion "0.1.0"
#define RepoRoot AddBackslash(SourcePath) + "..\.."

[Setup]
AppId={{37BAF0D8-1958-4926-A447-E7CE42D8E4A5}
AppName=Kraken Desktop
AppVersion={#AppVersion}
DefaultDirName={autopf}\Kraken Desktop
DefaultGroupName=Kraken
OutputDir={#RepoRoot}\dist\installers
OutputBaseFilename=KrakenDesktopSetup-{#AppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\KrakenDesktop.exe

[Files]
Source: "{#RepoRoot}\dist\windows\KrakenDesktop\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Kraken Desktop"; Filename: "{app}\KrakenDesktop.exe"
Name: "{autodesktop}\Kraken Desktop"; Filename: "{app}\KrakenDesktop.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"

[Run]
Filename: "{app}\KrakenDesktop.exe"; Description: "Запустить Kraken Desktop"; Flags: nowait postinstall skipifsilent
