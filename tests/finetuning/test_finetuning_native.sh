#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
finetuning_service_port=8015
ray_port=8265
service_name="finetuning"
TAG="comp"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build -t opea/finetuning:$TAG --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg HF_TOKEN=$HF_TOKEN -f comps/finetuning/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/finetuning built fail"
        exit 1
    else
        echo "opea/finetuning built successful"
    fi
}

function start_service() {
    export no_proxy="localhost,127.0.0.1,"${ip_address}
    export TAG=$TAG
    cd $WORKPATH/comps/finetuning/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 1m
}

function validate_upload() {
    local URL="$1"
    local SERVICE_NAME="$2"
    local DOCKER_NAME="$3"
    local EXPECTED_PURPOSE="$4"
    local EXPECTED_FILENAME="$5"

    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -F "file=@./$EXPECTED_FILENAME" -F purpose="$EXPECTED_PURPOSE" -H 'Content-Type: multipart/form-data' "$URL")
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')

    # Parse the JSON response
    purpose=$(echo "$RESPONSE_BODY" | jq -r '.purpose')
    filename=$(echo "$RESPONSE_BODY" | jq -r '.filename')

    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_upload_file.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."
    fi

    # Check if the parsed values match the expected values
    if [[ "$purpose" != "$EXPECTED_PURPOSE" || "$filename" != "$EXPECTED_FILENAME" ]]; then
        echo "[ $SERVICE_NAME ] Content does not match the expected result: $RESPONSE_BODY"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_upload_file.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] Content is as expected."
    fi

    sleep 10s
}

function validate_finetune() {
    local URL="$1"
    local SERVICE_NAME="$2"
    local DOCKER_NAME="$3"
    local EXPECTED_DATA="$4"
    local INPUT_DATA="$5"

    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -H 'Content-Type: application/json' -d "$INPUT_DATA" "$URL")
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')
    FINTUNING_ID=$(echo "$RESPONSE_BODY" | jq -r '.id')

    # Parse the JSON response
    purpose=$(echo "$RESPONSE_BODY" | jq -r '.purpose')
    filename=$(echo "$RESPONSE_BODY" | jq -r '.filename')

    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_create.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."
    fi

    # Check if the parsed values match the expected values
    if [[ "$RESPONSE_BODY" != *"$EXPECTED_DATA"* ]]; then
        echo "[ $SERVICE_NAME ] Content does not match the expected result: $RESPONSE_BODY"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_create.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] Content is as expected."
    fi

    sleep 10s

    # check finetuning job status
    URL="$URL/retrieve"
    for((i=1;i<=10;i++));
    do
	HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": "'$FINTUNING_ID'"}' "$URL")
	echo $HTTP_RESPONSE
	RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')
	STATUS=$(echo "$RESPONSE_BODY" | jq -r '.status')
	if [[ "$STATUS" == "succeeded" ]]; then
	    echo "training: succeeded."
	    break
	elif [[ "$STATUS" == "failed" ]]; then
	    echo "training: failed."
        docker logs $DOCKER_NAME 2>&1 | tee ${LOG_PATH}/finetuning-job.log
        RAY_JOB_ID=$(grep -o 'raysubmit_[A-Za-z0-9]\+' ${LOG_PATH}/finetuning-job.log | tail -n 1)
        echo "Log for failure."
        docker exec $DOCKER_NAME python -c "from ray.job_submission import JobSubmissionClient;client = JobSubmissionClient();print(client.get_job_logs('$RAY_JOB_ID'))"
	    exit 1
	else
	    echo "training: '$STATUS'"
	fi
	sleep 1m
    done
}

function validate_microservice() {
    cd $LOG_PATH
    export no_proxy="localhost,127.0.0.1,"${ip_address}

    ##########################
    #    general test         #
    ##########################
    # test /v1/dataprep upload file
    echo '[{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."},{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}]' > $LOG_PATH/test_data.json
    validate_upload \
        "http://${ip_address}:$finetuning_service_port/v1/files" \
        "general - upload" \
        "finetuning" \
        "fine-tune" \
        "test_data.json"

    # test /v1/fine_tuning/jobs
    validate_finetune \
        "http://${ip_address}:$finetuning_service_port/v1/fine_tuning/jobs" \
        "general - finetuning" \
        "finetuning" \
        '{"id":"ft-job' \
        '{"training_file": "test_data.json","model": "facebook/opt-125m"}'


    ##########################
    #    rerank test         #
    ##########################
    # test /v1/dataprep upload file
    cat <<EOF > test_data_rerank.json
{"query": "Five women walk along a beach wearing flip-flops.", "pos": ["Some women with flip-flops on, are walking along the beach"], "neg": ["The 4 women are sitting on the beach.", "There was a reform in 1996.", "She's not going to court to clear her record.", "The man is talking about hawaii.", "A woman is standing outside.", "The battle was over. ", "A group of people plays volleyball."]}
{"query": "A woman standing on a high cliff on one leg looking over a river.", "pos": ["A woman is standing on a cliff."], "neg": ["A woman sits on a chair.", "George Bush told the Republicans there was no way he would let them even consider this foolish idea, against his top advisors advice.", "The family was falling apart.", "no one showed up to the meeting", "A boy is sitting outside playing in the sand.", "Ended as soon as I received the wire.", "A child is reading in her bedroom."]}
{"query": "Two woman are playing instruments; one a clarinet, the other a violin.", "pos": ["Some people are playing a tune."], "neg": ["Two women are playing a guitar and drums.", "A man is skiing down a mountain.", "The fatal dose was not taken when the murderer thought it would be.", "Person on bike", "The girl is standing, leaning against the archway.", "A group of women watch soap operas.", "No matter how old people get they never forget. "]}
{"query": "A girl with a blue tank top sitting watching three dogs.", "pos": ["A girl is wearing blue."], "neg": ["A girl is with three cats.", "The people are watching a funeral procession.", "The child is wearing black.", "Financing is an issue for us in public schools.", "Kids at a pool.", "It is calming to be assaulted.", "I face a serious problem at eighteen years old. "]}
{"query": "A yellow dog running along a forest path.", "pos": ["a dog is running"], "neg": ["a cat is running", "Steele did not keep her original story.", "The rule discourages people to pay their child support.", "A man in a vest sits in a car.", "Person in black clothing, with white bandanna and sunglasses waits at a bus stop.", "Neither the Globe or Mail had comments on the current state of Canada's road system. ", "The Spring Creek facility is old and outdated."]}
{"query": "It sets out essential activities in each phase along with critical factors related to those activities.", "pos": ["Critical factors for essential activities are set out."], "neg": ["It lays out critical activities but makes no provision for critical factors related to those activities.", "People are assembled in protest.", "The state would prefer for you to do that.", "A girl sits beside a boy.", "Two males are performing.", "Nobody is jumping", "Conrad was being plotted against, to be hit on the head."]}
EOF
    validate_upload \
        "http://${ip_address}:$finetuning_service_port/v1/files" \
        "rerank - upload" \
        "finetuning" \
        "fine-tune" \
        "test_data_rerank.json"

    # test /v1/fine_tuning/jobs
    validate_finetune \
        "http://${ip_address}:$finetuning_service_port/v1/fine_tuning/jobs" \
        "rerank - finetuning" \
        "finetuning" \
        '{"id":"ft-job' \
        '{"training_file": "test_data_rerank.json","model": "BAAI/bge-reranker-base","General":{"task":"rerank","lora_config":null}}'


    ##########################
    #    llm pretrain test   #
    ##########################
    # test /v1/dataprep upload file
    cat <<EOF > test_data_pretrain.json
{"text": "Five women walk along a beach wearing flip-flops."}
{"text": "A woman standing on a high cliff on one leg looking over a river."}
{"text": "Two woman are playing instruments; one a clarinet, the other a violin."}
{"text": "A girl with a blue tank top sitting watching three dogs."}
{"text": "A yellow dog running along a forest path."}
{"text": "It sets out essential activities in each phase along with critical factors related to those activities."}
EOF
    validate_upload \
        "http://${ip_address}:$finetuning_service_port/v1/files" \
        "pretrain - upload" \
        "finetuning" \
        "fine-tune" \
        "test_data_pretrain.json"

    # test /v1/fine_tuning/jobs
    validate_finetune \
        "http://${ip_address}:$finetuning_service_port/v1/fine_tuning/jobs" \
        "pretrain - finetuning" \
        "finetuning" \
        '{"id":"ft-job' \
        '{"training_file": "test_data_pretrain.json","model": "facebook/opt-125m","General":{"task":"pretraining","lora_config":null}}'


    ##########################
    #    dpo test   #
    ##########################
    # test /v1/dataprep upload file
    cat <<EOF > test_data_dpo.jsonl
{"system": "", "question": "You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:", "chosen": "[\n  [\"AFC Ajax (amateurs)\", \"has ground\", \"Sportpark De Toekomst\"],\n  [\"Ajax Youth Academy\", \"plays at\", \"Sportpark De Toekomst\"]\n]", "rejected": " Sure, I'd be happy to help! Here are the RDF triplets for the input sentence:\n\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst]\n\nExplanation:\n\n* AFC Ajax (amateurs) is the subject of the first triplet, and hasGround is the predicate that describes the relationship between AFC Ajax (amateurs) and Sportpark De Toekomst.\n* Ajax Youth Academy is the subject of the second triplet, and playsAt is the predicate that describes the relationship between Ajax Youth Academy and Sportpark De Toekomst.\n\nNote that there may be other possible RDF triplets that could be derived from the input sentence, but the above triplets capture the main relationships present in the sentence."}
{"system": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.", "question": "Generate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One", "chosen": "Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One.", "rejected": " Sure! Here's a sentence that describes all the data you provided:\n\n\"Midsummer House is a moderately priced Chinese restaurant with a customer rating of 3 out of 5, located near All Bar One, offering a variety of delicious dishes.\""}
EOF
    validate_upload \
        "http://${ip_address}:$finetuning_service_port/v1/files" \
        "dpo - upload" \
        "finetuning" \
        "fine-tune" \
        "test_data_dpo.jsonl"

    # test /v1/fine_tuning/jobs
    validate_finetune \
        "http://${ip_address}:$finetuning_service_port/v1/fine_tuning/jobs" \
        "dpo - finetuning" \
        "finetuning" \
        '{"id":"ft-job' \
        '{"training_file": "test_data_dpo.jsonl","model": "facebook/opt-125m","General":{"task":"dpo"}}'

    ##########################
    #    reasoning SFT test  #
    ##########################
    # test /v1/dataprep upload file
    cat <<EOF > test_data_reasoning.jsonl
[
    {
        "Question": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?",
        "Complex_CoT": "Okay, let's see what's going on here. We've got sudden weakness in the person's left arm and leg - and that screams something neuro-related, maybe a stroke?\n\nBut wait, there's more. The right lower leg is swollen and tender, which is like waving a big flag for deep vein thrombosis, especially after a long flight or sitting around a lot.\n\nSo, now I'm thinking, how could a clot in the leg end up causing issues like weakness or stroke symptoms?\n\nOh, right! There's this thing called a paradoxical embolism. It can happen if there's some kind of short circuit in the heart - like a hole that shouldn't be there.\n\nLet's put this together: if a blood clot from the leg somehow travels to the left side of the heart, it could shoot off to the brain and cause that sudden weakness by blocking blood flow there.\n\nHmm, but how would the clot get from the right side of the heart to the left without going through the lungs and getting filtered out?\n\nHere's where our cardiac anomaly comes in: a patent foramen ovale or PFO. That's like a sneaky little shortcut in the heart between the right and left atria.\n\nAnd it's actually pretty common, found in about a quarter of adults, which definitely makes it the top suspect here.\n\nSo with all these pieces - long travel, leg clot, sudden weakness - a PFO fits the bill perfectly, letting a clot cross over and cause all this.\n\nEverything fits together pretty neatly, so I'd bet PFO is the heart issue waiting to be discovered. Yeah, that really clicks into place!",
        "Response": "The specific cardiac abnormality most likely to be found in this scenario is a patent foramen ovale (PFO). This condition could allow a blood clot from the venous system, such as one from a deep vein thrombosis in the leg, to bypass the lungs and pass directly into the arterial circulation. This can occur when the clot moves from the right atrium to the left atrium through the PFO. Once in the arterial system, the clot can travel to the brain, potentially causing an embolic stroke, which would explain the sudden weakness in the left arm and leg. The connection between the recent travel, which increases the risk of deep vein thrombosis, and the neurological symptoms suggests the presence of a PFO facilitating a paradoxical embolism."
    },
    {
        "Question": "A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110/min, respirations 22/min, and blood pressure 90/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured?",
        "Complex_CoT": "Okay, let's figure out what's going on here. A woman comes in with a stab wound from a screwdriver. It's in her chest, upper border of the 8th rib, left side, kind of around the midaxillary line. First thought, that's pretty close to where the lung sits, right?\n\nLet's talk about location first. This spot is along the left side of her body. Above the 8th rib, like that, is where a lot of important stuff lives, like the bottom part of the left lung, possibly the diaphragm too, especially considering how deep the screwdriver went.\n\nThe wound is 5 cm deep. That sounds pretty deep. I mean, it could definitely reach down to the lung tissue or maybe the diaphragm. Given that it's midaxillary, we're in the territory where the lower lobe of the left lung hangs out. It's also possible there's some intersection with where the diaphragm begins, but the lung feels more probable somehow.\n\nNow, her vitals are concerning: elevated heart rate and low blood pressure. This is serious. My gut says that this kind of vital sign picture could mean something like pneumothorax or maybe hemothorax. Both can happen if the lung gets punctured, and they can make the blood pressure drop and the heart rate skyrocket since she's obviously distressed.\n\nSo, putting it all together, the most obvious culprit is the lower lobe of the left lung. The wound's depth and her condition point that way. And, yeah, this adds up with pneumothorax or maybe blood collecting in the chest—the kind of stuff that can really mess with breathing and circulation.\n\nAlright, with this in mind, it sure seems like the lung is the most likely thing that got hurt here. Makes sense given the situation, where the wound is, considering her symptoms, and the whole setup.",
        "Response": "In this scenario, the most likely anatomical structure to be injured is the lower lobe of the left lung. The location of the stab wound—at the upper border of the 8th rib in the left midaxillary line—indicates proximity to the lower lobe of the lung. The depth of the wound (5 cm) suggests it is sufficient to reach lung tissue. Her vital signs of elevated heart rate and low blood pressure could signal complications like a pneumothorax or hemothorax, common consequences of lung trauma that would result from a penetrating injury in this area. Given these considerations, the lower lobe of the left lung is the most probable structure injured."
    }
]
EOF
    validate_upload \
        "http://${ip_address}:$finetuning_service_port/v1/files" \
        "dpo - upload" \
        "finetuning" \
        "fine-tune" \
        "test_data_reasoning.jsonl"

    # test /v1/fine_tuning/jobs
    validate_finetune \
        "http://${ip_address}:$finetuning_service_port/v1/fine_tuning/jobs" \
        "reasoning - finetuning" \
        "finetuning" \
        '{"id":"ft-job' \
        '{"training_file": "test_data_reasoning.jsonl","model": "Qwen/Qwen2.5-0.5B","General":{"task":"reasoning"}}'

}

function stop_docker() {
    cd $WORKPATH/comps/finetuning/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
