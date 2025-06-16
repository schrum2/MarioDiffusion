cd ..
REM Loop through all matching folders
for /D %%D in (Mar1and2-*) do (
    set "folder=%%~nxD"

    REM Use a regular expression-like split to extract prefix and number
    for /f "tokens=1,* delims=0123456789" %%A in ("!folder!") do (
        set "prefix=%%A"
        set "number_part=%%B"
    )

    REM Get numeric part from folder name
    set "suffix=!folder:%prefix%=!"

    for /f %%N in ("!suffix!") do (
        set /a num=%%N

        REM Track maximum number per prefix
        call set "existing=%%max_!prefix!%%"
        if not defined existing (
            set /a max_!prefix!=!num!
        ) else (
            if !num! GTR !max_!prefix!! (
                set /a max_!prefix!=!num!
            )
        )
    )
)

REM Loop over all collected prefixes and run the command
for /f "tokens=1,* delims==" %%K in ('set max_') do (
    set "prefix=%%K:max_=%%"
    set "END_NUM=%%L"
    echo Running: python calculate_execution_time.py !prefix! 0 !END_NUM!
    python calculate_execution_time.py !prefix! 0 !END_NUM!
)

endlocal