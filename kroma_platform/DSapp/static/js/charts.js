let typeChart, axisChart, organismChart;

function renderChart(ctx, label, data, type="pie") {
    return new Chart(ctx, {
        type: type,
        data: {
            labels: data.map(d => d[label]),
            datasets: [{
                label: 'Count',
                data: data.map(d => d.count),
                backgroundColor: ['#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#4472c4', '#a5a5a5', '#ff9999'],
            }]
        }
    });
}

function updateDashboard() {
    const params = new URLSearchParams({
        date_filter: $("#dateFilter").val(),
        ds_filter: $("#dsFilter").val(),
        organism_filter: $("#organismFilter").val(),
        start_date: $("#startDate").val(),
        end_date: $("#endDate").val(),
    });

    fetch(`/filter/?${params.toString()}`)
        .then(res => res.json())
        .then(data => {
            if (typeChart) typeChart.destroy();
            if (axisChart) axisChart.destroy();
            if (organismChart) organismChart.destroy();

            typeChart = renderChart(document.getElementById('typeChart'), 'type', data.type_data, 'pie');
            axisChart = renderChart(document.getElementById('axisChart'), 'axis', data.axis_data, 'bar');
            organismChart = renderChart(document.getElementById('organismChart'), 'organism', data.organism_data, 'bar');

            const tableBody = $("#articlesTable tbody");
            tableBody.empty();
            data.table_data.forEach(article => {
                tableBody.append(`<tr data-id="${article.pmcid}"><td>${article.title}</td></tr>`);
            });

            $("#articlesTable").DataTable();
        });
}

$("#applyFilters").on("click", updateDashboard);

$(document).on("click", "#articlesTable tbody tr", function () {
    const pmcid = $(this).data("id");
    fetch(`/article/${pmcid}/`)
        .then(res => res.json())
        .then(article => {
            $("#articleDetail").html(`
                <h3>${article.journal}</h3>
                <p><b>Authors:</b> ${article.authors}</p>
                <p><b>Organism:</b> ${article.organism}</p>
                <p><b>Date:</b> ${article.date}</p>
                <p><b>Abstract:</b> ${article.abstract}</p>
                <p><a href="${article.url}" target="_blank">View on PubMed</a></p>
                <button onclick="window.open('/pdf/${pmcid}/')">View PDF</button>
            `).removeClass("hidden");
        });
});

// Initial load
updateDashboard();
