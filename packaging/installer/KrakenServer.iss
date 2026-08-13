#define AppVersion "0.1.0"
#define RepoRoot AddBackslash(SourcePath) + "..\.."

[Setup]
AppId={{A967C65D-AD95-4B44-A628-C60214E71261}
AppName=Kraken Server
AppVersion={#AppVersion}
DefaultDirName={autopf}\Kraken Server
DefaultGroupName=Kraken
OutputDir={#RepoRoot}\dist\installers
OutputBaseFilename=KrakenServerSetup-{#AppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\KrakenServer.exe

[Files]
Source: "{#RepoRoot}\dist\windows\KrakenServer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Настройка Kraken Server"; Filename: "{app}\KrakenAdmin.exe"; Parameters: "setup-server --install-service --server-executable ""{app}\KrakenServer.exe"""; WorkingDir: "{app}"

[Run]
Filename: "{app}\KrakenAdmin.exe"; Parameters: "setup-server --install-service --server-executable ""{app}\KrakenServer.exe"""; Description: "Настроить базу данных и первого администратора"; Flags: postinstall waituntilterminated skipifsilent

[UninstallRun]
Filename: "{sys}\sc.exe"; Parameters: "stop KrakenServer"; Flags: runhidden waituntilterminated; RunOnceId: "StopKrakenServer"
Filename: "{sys}\sc.exe"; Parameters: "delete KrakenServer"; Flags: runhidden waituntilterminated; RunOnceId: "DeleteKrakenServer"
