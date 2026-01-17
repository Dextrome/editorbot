$runTime = (Get-Date).AddHours(3)
$timeStr = $runTime.ToString("HH:mm")
$dateStr = $runTime.ToString("yyyy-MM-dd")

Write-Host "Scheduling training for $dateStr at $timeStr"

schtasks /create /tn "PointerNetworkTraining" /tr "F:\editorbot\scripts\run_training.bat" /sc once /st $timeStr /sd $dateStr /f

if ($LASTEXITCODE -eq 0) {
    Write-Host "Task scheduled successfully!"
} else {
    Write-Host "Trying alternative date format..."
    $dateStr2 = $runTime.ToString("dd/MM/yyyy")
    schtasks /create /tn "PointerNetworkTraining" /tr "F:\editorbot\scripts\run_training.bat" /sc once /st $timeStr /sd $dateStr2 /f
}
