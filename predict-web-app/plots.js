// default settings
Chart.defaults.global.defaultFontColor = 'white';
Chart.defaults.global.defaultFontSize = 14;

// plot grid colors
gridLines = {
    color: 'rgba(255, 255, 255, 0.1)'
};

// scores barchart settings
var barchart = new Chart(document.getElementById('barchart').getContext('2d'), {
    type: 'bar',
    data: {
        labels: ["Falling", "Jumping", "Lying", "Running", "Sitting", "Standing", "Walking"],
        datasets: [{
            label: "predictions",
            backgroundColor: 'rgb(230, 172, 46)',
            borderColor: 'rgb(230, 172, 46)',
            data: new Array(7).fill(0.01),
        }]
    },
    options: {
        legend: { display: false },
        scales: {
            xAxes: [{ gridLines: gridLines }],
            yAxes: [{
                gridLines: gridLines,
                ticks: { min: 0, max: 1 },
            }]
        }
    }
});

lineoptions = {
    maintainAspectRatio: false, // needed to resize chart with css
    legend: { position: 'right' },
    animation: {
        duration: 40 // set to 0 to boost performance
    },
    scales: {
        yAxes: [{
            ticks: { suggestedMin: -5, suggestedMax: 5 },
            gridLines: gridLines
        }],
        xAxes: [{ display: false }],
    },
    /* to boost performance
    elements: {
        line: {
            tension: 0, // disables bezier curves
        }
    }
    */
}

// number of points to show in the acc, gyro plots
pointsnr = 50;
var accchart = new Chart(document.getElementById('accchart').getContext('2d'), {
    type: 'line',
    data: {
        labels: new Array(pointsnr).fill(0),
        datasets: [{
            label: "acc_x",
            borderColor: 'rgb(8, 227, 169)',
            data: new Array(pointsnr).fill(0),
            fill: false,
            pointRadius: 0 // don't show the points
        },
        {
            label: "acc_y",
            borderColor: 'rgb(74, 154, 217)',
            data: new Array(pointsnr).fill(0.1),
            fill: false,
            pointRadius: 0
        },
        {
            label: "acc_z",
            borderColor: 'rgb(186, 21, 221)',
            data: new Array(pointsnr).fill(0.2),
            fill: false,
            pointRadius: 0
        }]
    },
    options: lineoptions
});
var gyrochart = new Chart(document.getElementById('gyrochart').getContext('2d'), {
    type: 'line',
    data: {
        labels: new Array(pointsnr).fill(0),
        datasets: [{
            label: "gyr_x",
            borderColor: 'rgb(8, 227, 169)',
            data: new Array(pointsnr).fill(0),
            fill: false,
            pointRadius: 0
        },
        {
            label: "gyr_y",
            borderColor: 'rgb(74, 154, 217)',
            data: new Array(pointsnr).fill(0.1),
            fill: false,
            pointRadius: 0
        },
        {
            label: "gyr_z",
            borderColor: 'rgb(186, 21, 221)',
            data: new Array(pointsnr).fill(0.2),
            fill: false,
            pointRadius: 0
        }]
    },
    options: lineoptions
});

// websocket address must be equal to the one set on server/server.py
var ws = new WebSocket("ws://127.0.0.1:5678/");
ws.onmessage = function (event) {
    data = JSON.parse(event.data);

    // update acc plot
    accchart.data.datasets.forEach((dataset, i) => {
        // remove last point
        dataset.data.shift();
        // add the new point
        // acc values are from position 2 to 4
        dataset.data.push(data.realtime[2+i]);
    });
    accchart.update();

    // update gyro plot
    gyrochart.data.datasets.forEach((dataset, j) => {
        dataset.data.shift();
        dataset.data.push(data.realtime[6+j]);
    });
    gyrochart.update();

    // update the scores, if a prediction has been made
    if (data.scores.length != 0){
        barchart.data.datasets.forEach((dataset) => {
            dataset.data = data.scores[0];
        });
        barchart.update();
        // update prediction string
        document.getElementById("prediction").textContent = data.prediction;
    }
    // update framerate string
    document.getElementById("framerate").textContent = data.framerate;
};