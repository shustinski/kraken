; Karakal Inno Setup installer.
; AppId is permanent — never change it (beta and stable share one install).
#define AppVersion "0.1.0-beta0"
#ifndef AppVersionNumeric
  #define AppVersionNumeric "0.1.0.0"
#endif
#define RepoRoot AddBackslash(SourcePath) + "..\..\.."
#define KarakalRoot AddBackslash(SourcePath) + ".."

#ifexist "version_override.iss"
  #include "version_override.iss"
#endif

[Setup]
AppId={{9FCD1093-438E-4B62-945B-3BF35E841C81}
AppName=Karakal
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersionNumeric}
DefaultDirName={localappdata}\Programs\Karakal
DefaultGroupName=Karakal
OutputDir={#KarakalRoot}\dist\installers
OutputBaseFilename=Karakal-setup-{#AppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
UninstallDisplayIcon={app}\karakal.exe
DisableProgramGroupPage=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; Flags: unchecked

[Dirs]
; Runtime data created by the app — keep across updates/uninstall of app files.
Name: "{app}\logs"; Flags: uninsneveruninstall

[Files]
; settings.ini and logs\ are created at runtime and are NOT in the build tree,
; so Inno leaves them in place on upgrade (no InstallDelete for them).
Source: "{#KarakalRoot}\dist\windows\karakal\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Karakal"; Filename: "{app}\karakal.exe"
Name: "{autodesktop}\Karakal"; Filename: "{app}\karakal.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\karakal.exe"; Flags: nowait postinstall skipifnotsilent
Filename: "{app}\karakal.exe"; Description: "Запустить Karakal"; Flags: nowait postinstall skipifsilent

[Code]
var
  UpdateRootValue: String;

function GetCommandlineParam(ParamName: String): String;
var
  I: Integer;
  S: String;
  Prefix: String;
begin
  Result := '';
  Prefix := '/' + UpperCase(ParamName) + '=';
  for I := 1 to ParamCount do
  begin
    S := ParamStr(I);
    if Pos(Prefix, UpperCase(S)) = 1 then
    begin
      Result := Copy(S, Length(Prefix) + 1, MaxInt);
      // Strip surrounding quotes if present
      if (Length(Result) >= 2) and (Result[1] = '"') and (Result[Length(Result)] = '"') then
        Result := Copy(Result, 2, Length(Result) - 2);
      Exit;
    end;
  end;
end;

function InitializeSetup(): Boolean;
begin
  UpdateRootValue := GetCommandlineParam('UPDATEROOT');
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  IniPath: String;
begin
  if CurStep = ssPostInstall then
  begin
    if UpdateRootValue <> '' then
    begin
      IniPath := ExpandConstant('{app}\settings.ini');
      SetIniString('General', 'update_root', UpdateRootValue, IniPath);
    end;
  end;
end;
