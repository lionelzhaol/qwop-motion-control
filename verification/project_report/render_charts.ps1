param([string]$PlanPath,[string]$OutputDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$plans=Get-Content -LiteralPath $PlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$index=0
foreach($plan in $plans) {
    $chart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width=1440
    $chart.Height=920
    $chart.BackColor=[Drawing.ColorTranslator]::FromHtml('#F7F9FC')
    $chart.AntiAliasing='All'
    $chart.TextAntiAliasingQuality='High'
    $chart.Font=New-Object Drawing.Font('Microsoft YaHei',9)
    $heading=New-Object System.Windows.Forms.DataVisualization.Charting.Title($plan.title)
    $heading.Font=New-Object Drawing.Font('Microsoft YaHei',15,[Drawing.FontStyle]::Bold)
    $heading.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(3,1,94,4.5)
    $chart.Titles.Add($heading)
    $subtitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title($plan.subtitle)
    $subtitle.Font=New-Object Drawing.Font('Microsoft YaHei',9)
    $subtitle.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(3,5.5,94,3)
    $chart.Titles.Add($subtitle)
    $positions=@(@(2,15,46,34),@(51,15,46,34),@(2,59,46,34),@(51,59,46,34))
    for($i=0;$i -lt $plan.panels.Count;$i++) {
        $panel=$plan.panels[$i]
        $pos=$positions[$i]
        $area=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('area'+$i)
        $area.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($pos[0],$pos[1],$pos[2],$pos[3])
        $area.BackColor=[Drawing.Color]::White
        $area.AxisX.Title=$plan.xlabel
        $area.AxisY.Title=$panel.ylabel
        $area.AxisX.LabelStyle.Format='0.##'
        $area.AxisY.LabelStyle.Format='0.###'
        $area.AxisX.IsStartedFromZero=$true
        $area.AxisX.Minimum=0
        if($panel.labels) {
            $area.AxisX.Minimum=0.5
            $area.AxisX.Maximum=$panel.labels.Count+0.5
            $area.AxisX.Interval=1
            $area.AxisY.Minimum=0
            for($labelIndex=0;$labelIndex -lt $panel.labels.Count;$labelIndex++) {
                $area.AxisX.CustomLabels.Add(($labelIndex+0.5),($labelIndex+1.5),$panel.labels[$labelIndex]) | Out-Null
            }
        }
        $area.AxisY.IsStartedFromZero=$false
        foreach($axis in @($area.AxisX,$area.AxisY,$area.AxisY2)) {
            $axis.TitleFont=New-Object Drawing.Font('Microsoft YaHei',9)
            $axis.LabelStyle.Font=New-Object Drawing.Font('Microsoft YaHei',8)
            $axis.LineColor=[Drawing.Color]::FromArgb(180,190,200)
            $axis.MajorGrid.LineColor=[Drawing.Color]::FromArgb(231,235,240)
            $axis.MajorTickMark.LineColor=[Drawing.Color]::FromArgb(180,190,200)
        }
        if($panel.ylabel2){$area.AxisY2.Title=$panel.ylabel2;$area.AxisY2.MajorGrid.Enabled=$false}
        $chart.ChartAreas.Add($area)
        $title=New-Object System.Windows.Forms.DataVisualization.Charting.Title($panel.title)
        $title.Font=New-Object Drawing.Font('Microsoft YaHei',10,[Drawing.FontStyle]::Bold)
        $title.Alignment='MiddleLeft'
        $title.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(($pos[0]+2),($pos[1]-5),44,4)
        $chart.Titles.Add($title)
        $legend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend('legend'+$i)
        $legend.DockedToChartArea=$area.Name
        $legend.Docking='Top'
        $legend.IsDockedInsideChartArea=$true
        $legend.BackColor=[Drawing.Color]::Transparent
        $legend.Font=New-Object Drawing.Font('Microsoft YaHei',8)
        $chart.Legends.Add($legend)
        $j=0
        foreach($row in $panel.series) {
            if($row.points.Count -eq 0){continue}
            $line=New-Object System.Windows.Forms.DataVisualization.Charting.Series("s${i}_${j}")
            $line.ChartArea=$area.Name
            $line.Legend=$legend.Name
            $line.LegendText=$row.name
            $line.ChartType='Line'
            if($row.chart_type){$line.ChartType=$row.chart_type}
            if($row.chart_type -eq 'Column') {
                $line.IsValueShownAsLabel=$true
                $line.LabelFormat='0.##'
                $line.Font=New-Object Drawing.Font('Microsoft YaHei',9)
                $line.IsVisibleInLegend=$false
            }
            $line.BorderWidth=2
            $line.Color=[Drawing.ColorTranslator]::FromHtml($row.color)
            if($row.secondary){$line.YAxisType='Secondary'}
            if($row.points.Count -lt 80){$line.MarkerStyle='Circle';$line.MarkerSize=4}
            foreach($p in $row.points){$line.Points.AddXY([double]$p[0],[double]$p[1]) | Out-Null}
            $chart.Series.Add($line)
            $j++
        }
        if($j -eq 0) {
            $note=New-Object System.Windows.Forms.DataVisualization.Charting.TextAnnotation
            $note.Text='此项没有可用标量，不补造数据'
            $note.Font=New-Object Drawing.Font('Microsoft YaHei',11)
            $note.X=$pos[0]+3
            $note.Y=$pos[1]+12
            $note.Width=40
            $note.Height=8
            $chart.Annotations.Add($note)
        }
    }
    $destination=Join-Path $OutputDirectory $plan.file
    $chart.SaveImage($destination,'Png')
    $chart.Dispose()
    $index++
    if($index % 10 -eq 0){Write-Output "Rendered $index / $($plans.Count) charts"}
}
Write-Output "Completed $index charts"
