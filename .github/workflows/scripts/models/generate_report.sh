#!/bin/bash
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x
WORKSPACE=generated
last_log_path=FinalReport
summaryLog=${WORKSPACE}/summary.log
summaryLogLast=${last_log_path}/summary.log
PATTERN='[-a-zA-Z0-9_]*='

function main {
    echo "summaryLog: ${summaryLog}"
    echo "summaryLogLast: ${summaryLogLast}"
    echo "is_perf_reg=false" >> "$GITHUB_ENV"
    preprocessing
    generate_html_head
    generate_html_overview
    generate_results
    generate_html_footer
}

function preprocessing {
    for file_path in log/*
    do
        if [[ -d ${file_path} ]] && [[ -f ${file_path}/summary.log ]]; then
            cat ${file_path}/summary.log >> ${summaryLog}
        fi
    done
}

function generate_html_overview {
    Test_Info_Title="<th colspan="4">Test Branch</th> <th colspan="4">Commit ID</th> "
    Test_Info="<th colspan="4">${MR_source_branch}</th> <th colspan="4">${ghprbActualCommit}</th> "

    cat >>${WORKSPACE}/report.html <<eof

<body>
    <div id="main">
        <h1 align="center">ITREX Tests
        [ <a href="${RUN_DISPLAY_URL}">Job-${BUILD_NUMBER}</a> ]</h1>
      <h1 align="center">Test Status: ${JOB_STATUS}</h1>
        <h2>Summary</h2>
        <table class="features-table">
            <tr>
              <th>Repo</th>
              ${Test_Info_Title}
              </tr>
              <tr>
                    <td><a href="https://github.com/intel/intel-extension-for-transformers">ITREX</a></td>
              ${Test_Info}
                </tr>
        </table>
eof
}

function generate_results {
    cat >>${WORKSPACE}/report.html <<eof
    <h2>Performance</h2>
      <table class="features-table">
        <tr>
          <th>Device</th>
          <th>Tasks</th>
          <th>Model</th>
          <th>Datasets</th>
          <th>VS</th>
          <th>Accuracy</th>
        </tr>
eof

    devices=$(cat ${summaryLog} | cut -d';' -f1 | awk '!a[$0]++')
    for device in ${devices[@]}; do
        models=$(cat ${summaryLog} | grep "${device};" | cut -d';' -f2 | awk '!a[$0]++')
        for model in ${models[@]}; do
            tasks=$(cat ${summaryLog} | grep "${device};${model};" | cut -d';' -f3 | awk '!a[$0]++')
            for task in ${tasks[@]}; do
                datasets=$(cat ${summaryLog} | grep "${device};${model};${task};" | cut -d';' -f4 | awk '!a[$0]++')
                for dataset in ${datasets[@]}; do
                    benchmark_pattern="${device};${model};${task};${dataset};"
                    acc=$(cat ${summaryLog} | grep "${benchmark_pattern}" | cut -d';' -f5 | awk '!a[$0]++')
                    acc_last=nan
                    if [ $(cat ${summaryLogLast} | grep -c "${benchmark_pattern}") != 0 ]; then
                        acc_last=$(cat ${summaryLogLast} | grep "${benchmark_pattern}" | cut -d';' -f5 | awk '!a[$0]++')
                    fi
                    generate_core
                done
            done
        done
    done
    cat >>${WORKSPACE}/report.html <<eof
    </table>
eof
}

function generate_core {
    echo "<tr><td rowspan=3>${device}</td><td rowspan=3>${model}</td><td rowspan=3>${task}</td><td rowspan=3>${dataset}</td><td>New</td>" >>${WORKSPACE}/report.html
    echo | awk -v acc=${acc} -v acc_l=${acc_last} '
        function show_benchmark(a) {
            if(a ~/[1-9]/) {
                printf("<td>%.2f</td>\n",a);
            }else {
                printf("<td></td>\n");
            }
        }
        function compare_new_last(a,b){
            if(a ~/[1-9]/ && b ~/[1-9]/) {
                target = b / a;
                if(target >= 0.945) {
                    status_png = "background-color:#90EE90";
                }else {
                    status_png = "background-color:#FFD2D2";
                    job_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            }else{
                if(a == ""){
                    job_status = "fail"
                    status_png = "background-color:#FFD2D2";
                    printf("<td style=\"%s\"></td>", status_png);
                }else{
                    printf("<td class=\"col-cell col-cell3\"></td>");
                }
            }
        }
        BEGIN {
            job_status = "pass"
        }{
            // current
            show_benchmark(acc)
            // Last
            printf("</tr>\n<tr><td>Last</td>")
            show_benchmark(acc_l)
            // current vs last
            printf("</tr>\n<tr><td>New/Last</td>");
            compare_new_last(acc,acc_l)
            printf("</tr>\n");
        } END{
          printf("\n%s", job_status);
        }
    ' >>${WORKSPACE}/report.html
    job_state=$(tail -1 ${WORKSPACE}/report.html)
    sed -i '$s/.*//' ${WORKSPACE}/report.html
    if [ ${job_state} == 'fail' ]; then
        echo "is_perf_reg=true" >> "$GITHUB_ENV"
    fi
}

function generate_html_head {
    cat >${WORKSPACE}/report.html <<eof
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Tests - TensorFlow - Jenkins</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: white no-repeat left top;
        }

        #main {
            // width: 100%;
            margin: 20px auto 10px auto;
            background: white;
            -moz-border-radius: 8px;
            -webkit-border-radius: 8px;
            padding: 0 30px 30px 30px;
            border: 1px solid #adaa9f;
            -moz-box-shadow: 0 2px 2px #9c9c9c;
            -webkit-box-shadow: 0 2px 2px #9c9c9c;
        }

        .features-table {
            width: 100%;
            margin: 0 auto;
            border-collapse: separate;
            border-spacing: 0;
            text-shadow: 0 1px 0 #fff;
            color: #2a2a2a;
            background: #fafafa;
            background-image: -moz-linear-gradient(top, #fff, #eaeaea, #fff);
            /* Firefox 3.6 */
            background-image: -webkit-gradient(linear, center bottom, center top, from(#fff), color-stop(0.5, #eaeaea), to(#fff));
            font-family: Verdana, Arial, Helvetica
        }

        .features-table th,
        td {
            text-align: center;
            height: 25px;
            line-height: 25px;
            padding: 0 8px;
            border: 1px solid #cdcdcd;
            box-shadow: 0 1px 0 white;
            -moz-box-shadow: 0 1px 0 white;
            -webkit-box-shadow: 0 1px 0 white;
            white-space: nowrap;
        }

        .no-border th {
            box-shadow: none;
            -moz-box-shadow: none;
            -webkit-box-shadow: none;
        }

        .col-cell {
            text-align: center;
            width: 150px;
            font: normal 1em Verdana, Arial, Helvetica;
        }

        .col-cell3 {
            background: #efefef;
            background: rgba(144, 144, 144, 0.15);
        }

        .col-cell1,
        .col-cell2 {
            background: #B0C4DE;
            background: rgba(176, 196, 222, 0.3);
        }

        .col-cellh {
            font: bold 1.3em 'trebuchet MS', 'Lucida Sans', Arial;
            -moz-border-radius-topright: 10px;
            -moz-border-radius-topleft: 10px;
            border-top-right-radius: 10px;
            border-top-left-radius: 10px;
            border-top: 1px solid #eaeaea !important;
        }

        .col-cellf {
            font: bold 1.4em Georgia;
            -moz-border-radius-bottomright: 10px;
            -moz-border-radius-bottomleft: 10px;
            border-bottom-right-radius: 10px;
            border-bottom-left-radius: 10px;
            border-bottom: 1px solid #dadada !important;
        }
    </style>
</head>
eof
}

function generate_html_footer {
    cat >>${WORKSPACE}/report.html <<eof
    </div>
</body>
</html>
eof
}

main
