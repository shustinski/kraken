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

[CustomMessages]
russian.UpdateRootCaption=Папка обновлений
russian.UpdateRootDescription=Откуда Каракал будет брать новые версии
russian.UpdateRootSubCaption=Укажите сетевую папку с обновлениями Каракала (в ней лежат папки beta и stable), например \\СЕРВЕР\Karakal.%nПри следующих запусках программа сама найдёт там новую версию и предложит её поставить.%nОставьте поле пустым, если папка уже задана в сборке.
russian.UpdateRootLabel=Папка обновлений:
russian.UpdateRootBrowse=Обзор…
russian.UpdateRootBrowseTitle=Выберите папку обновлений Каракала
russian.UpdateRootMissing=Папка %1 сейчас недоступна или в ней нет beta\version.json.%n%nВсё равно использовать её? Обновления заработают, когда папка станет доступна.
english.UpdateRootCaption=Update folder
english.UpdateRootDescription=Where Karakal takes new versions from
english.UpdateRootSubCaption=Enter the network folder with Karakal updates (it holds the beta and stable folders), e.g. \\SERVER\Karakal.%nOn later starts the program finds new versions there and offers to install them.%nLeave it empty if the folder is already set in the build.
english.UpdateRootLabel=Update folder:
english.UpdateRootBrowse=Browse…
english.UpdateRootBrowseTitle=Select the Karakal update folder
english.UpdateRootMissing=Folder %1 is not reachable now or has no beta\version.json.%n%nUse it anyway? Updates start working once the folder is reachable.

[Code]
var
  UpdateRootValue: String;
  UpdateRootPage: TInputQueryWizardPage;


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

// settings.ini is read by Qt, which treats a backslash as an escape: G:\ProgramStore came back
// as G:rogramStore. Forward slashes are plain characters there and Windows accepts them in paths.
function IniPath(Folder: String): String;
begin
  Result := Folder;
  StringChangeEx(Result, '\', '/', True);
end;

function ShownPath(Folder: String): String;
begin
  Result := Folder;
  StringChangeEx(Result, '/', '\', True);
end;

// The update folder already written by a previous install, if it still exists.
// A missing one is not offered: it may be a path Qt already damaged.
function InstalledUpdateRoot(): String;
begin
  Result := ShownPath(GetIniString('General', 'update_root', '', AddBackslash(WizardDirValue) + 'settings.ini'));
  if (Result <> '') and not DirExists(Result) then
    Result := '';
end;

procedure UpdateRootBrowseClick(Sender: TObject);
var
  Folder: String;
begin
  Folder := Trim(UpdateRootPage.Values[0]);
  if BrowseForFolder(CustomMessage('UpdateRootBrowseTitle'), Folder, False) then
    UpdateRootPage.Values[0] := Folder;
end;

procedure InitializeWizard();
var
  Edit: TPasswordEdit;
  Browse: TNewButton;
begin
  // Page after the install folder: the network folder this machine updates from.
  UpdateRootPage := CreateInputQueryPage(wpSelectDir,
    CustomMessage('UpdateRootCaption'), CustomMessage('UpdateRootDescription'),
    CustomMessage('UpdateRootSubCaption'));
  UpdateRootPage.Add(CustomMessage('UpdateRootLabel'), False);
  UpdateRootPage.Values[0] := UpdateRootValue;
  Edit := UpdateRootPage.Edits[0];
  Edit.Width := Edit.Width - ScaleX(96);
  Browse := TNewButton.Create(UpdateRootPage);
  Browse.Parent := UpdateRootPage.Surface;
  Browse.Caption := CustomMessage('UpdateRootBrowse');
  Browse.Left := Edit.Left + Edit.Width + ScaleX(8);
  Browse.Top := Edit.Top - ScaleY(1);
  Browse.Width := ScaleX(88);
  Browse.Height := Edit.Height + ScaleY(2);
  Browse.OnClick := @UpdateRootBrowseClick;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  // On a reinstall show the folder this machine already uses.
  if (CurPageID = UpdateRootPage.ID) and (Trim(UpdateRootPage.Values[0]) = '') then
    UpdateRootPage.Values[0] := InstalledUpdateRoot();
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Folder: String;
begin
  Result := True;
  if WizardSilent or (CurPageID <> UpdateRootPage.ID) then
    Exit;
  Folder := Trim(UpdateRootPage.Values[0]);
  if (Folder <> '') and not FileExists(AddBackslash(Folder) + 'beta\version.json') then
    Result := MsgBox(FmtMessage(CustomMessage('UpdateRootMissing'), [Folder]),
      mbConfirmation, MB_YESNO) = IDYES;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Folder: String;
begin
  if CurStep = ssPostInstall then
  begin
    // Empty field keeps whatever the machine had (an update never clears the folder).
    Folder := Trim(UpdateRootPage.Values[0]);
    if Folder <> '' then
      SetIniString('General', 'update_root', IniPath(Folder), ExpandConstant('{app}\settings.ini'));
  end;
end;
