pipeline {
    agent any

    environment {
        APP_DIR       = "/opt/kidney-disease-streamlit"
        VENV_DIR      = "${APP_DIR}/venv"
        APP_PORT      = "8501"
        PID_FILE      = "${APP_DIR}/streamlit.pid"
        LOG_FILE      = "${APP_DIR}/streamlit.log"
    }

    stages {

        stage('Checkout') {
            steps {
                echo '📥 Checking out source code...'
                checkout scm
            }
        }

        stage('Whoami Check') {
            steps {
                sh 'whoami'
                sh 'id'
            }
        }

        stage('Prepare App Directory') {
            steps {
                echo '📁 Preparing deployment directory...'
                sh '''
                    mkdir -p ${APP_DIR}
                    cp -r ./* ${APP_DIR}/
                '''
            }
        }

        stage('Python Setup') {
            steps {
                echo '🐍 Setting up Python environment...'
                sh '''
                    cd ${APP_DIR}
                    python3 --version
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Test') {
            steps {
                echo '🧪 Running basic checks...'
                sh '''
                    cd ${APP_DIR}
                    . venv/bin/activate
                    python -m py_compile app.py
                    echo "Python syntax check passed!"
                '''
            }
        }

        stage('Stop Existing App') {
            steps {
                echo '🛑 Stopping any existing running app...'
                sh '''
                    if [ -f ${PID_FILE} ]; then
                        OLD_PID=$(cat ${PID_FILE})
                        if ps -p $OLD_PID > /dev/null 2>&1; then
                            kill $OLD_PID
                            sleep 2
                            echo "Old process $OLD_PID stopped"
                        fi
                        rm -f ${PID_FILE}
                    fi
                    fuser -k ${APP_PORT}/tcp || true
                '''
            }
        }

        stage('Deploy') {
            steps {
                echo '🚀 Starting Streamlit application...'
                sh '''
                    cd ${APP_DIR}
                    . venv/bin/activate

                    nohup streamlit run app.py \
                        --server.address=0.0.0.0 \
                        --server.port=${APP_PORT} \
                        --server.headless=true \
                        > ${LOG_FILE} 2>&1 &

                    echo $! > ${PID_FILE}
                    sleep 3
                    echo "Started with PID $(cat ${PID_FILE})"
                '''
            }
        }

        stage('Health Check') {
            steps {
                echo '❤️ Checking Streamlit application...'
                sh '''
                    sleep 8
                    curl -f http://localhost:${APP_PORT}/_stcore/health
                    echo ""
                    echo "✅ Streamlit application is running!"
                '''
            }
        }
    }

    post {
        success {
            echo 'PIPELINE SUCCESSFUL - App running at http://<server-ip>:8501'
        }
        failure {
            echo 'PIPELINE FAILED — check console output'
            sh 'cat ${LOG_FILE} || true'
        }
    }
}
