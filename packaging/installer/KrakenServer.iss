#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define RepoRoot AddBackslash(SourcePath) + "..\.."
#ifndef BuildRoot
  #define BuildRoot RepoRoot + "\dist\windows"
#endif

[Setup]
#ifdef AuditAppId
AppId={#AuditAppId}
#else
AppId={{A967C65D-AD95-4B44-A628-C60214E71261}
#endif
AppName=Kraken Server
AppVersion={#AppVersion}
DefaultDirName={autopf}\Kraken Server
#ifdef AuditAppId
DefaultGroupName={#AuditAppId}
#else
DefaultGroupName=Kraken
#endif
OutputDir={#RepoRoot}\dist\installers
OutputBaseFilename=KrakenServerSetup-{#AppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#ifdef AuditAppId
PrivilegesRequired=lowest
#else
PrivilegesRequired=admin
#endif
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\KrakenServer.exe
CloseApplications=no
RestartApplications=no

[Files]
Source: "{#BuildRoot}\KrakenServer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Настроить Kraken Server"; Filename: "{app}\KrakenAdmin.exe"; Parameters: "init --config ""{commonappdata}\Kraken\Server\server.toml"" --blob-root ""{commonappdata}\Kraken\Server\blobs"" --install-service --server-executable ""{app}\KrakenServer.exe"""; WorkingDir: "{app}"
Name: "{group}\Проверить Kraken Server"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\Test-KrakenLocal.ps1"" -Config ""{commonappdata}\Kraken\Server\server.toml"""; WorkingDir: "{app}"

[Run]
Filename: "{app}\KrakenAdmin.exe"; Parameters: "init --config ""{commonappdata}\Kraken\Server\server.toml"" --blob-root ""{commonappdata}\Kraken\Server\blobs"" --install-service --server-executable ""{app}\KrakenServer.exe"""; Description: "Создать базу, настроить сервер и первого администратора"; Flags: postinstall waituntilterminated skipifsilent

[UninstallRun]
Filename: "{sys}\sc.exe"; Parameters: "stop KrakenServer"; Flags: runhidden waituntilterminated; RunOnceId: "StopKrakenServer"; Check: OwnsServerService
Filename: "{sys}\sc.exe"; Parameters: "delete KrakenServer"; Flags: runhidden waituntilterminated; RunOnceId: "DeleteKrakenServer"; Check: OwnsServerService

[Code]
#include "DeploymentGuard.iss"

function OwnsServerService: Boolean;
var
  Command, Executable: String;
  ClosingQuote, Space: Integer;
begin
  Result := False;
  if not RegQueryStringValue(HKLM64, 'SYSTEM\CurrentControlSet\Services\KrakenServer', 'ImagePath', Command) then
    Exit;
  Command := Trim(Command);
  if Copy(Command, 1, 1) = '"' then
  begin
    Delete(Command, 1, 1);
    ClosingQuote := Pos('"', Command);
    if ClosingQuote = 0 then Exit;
    Executable := Copy(Command, 1, ClosingQuote - 1);
    if (Length(Command) > ClosingQuote) and (Copy(Command, ClosingQuote + 1, 1) <> ' ') then Exit;
  end
  else
  begin
    Space := Pos(' ', Command);
    if Space = 0 then Executable := Command
    else Executable := Copy(Command, 1, Space - 1);
  end;
  Result := CompareText(ExpandFileName(Executable), ExpandConstant('{app}\KrakenServer.exe')) = 0;
  if not Result then Log('Preserving KrakenServer service: executable belongs to another installation.');
end;
