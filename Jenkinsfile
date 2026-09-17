pipeline {
    agent any

    environment {
        IMAGE_REPO = "naim8855/kidney-disease-app"
        IMAGE_TAG = "v${BUILD_NUMBER}"
        CONTAINER_NAME = "kidney-container"
        APP_PORT = "8501"
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                sh '''
                    docker build -t $IMAGE_REPO:$IMAGE_TAG -t $IMAGE_REPO:latest .
                '''
            }
        }

        stage('Login to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                }
            }
        }

        stage('Push Image') {
            steps {
                sh '''
                    docker push $IMAGE_REPO:$IMAGE_TAG
                    docker push $IMAGE_REPO:latest
                '''
            }
        }

        stage('Deploy App') {
            steps {
                sh '''
                    docker rm -f $CONTAINER_NAME || true
                    docker run -d \
                      --name $CONTAINER_NAME \
                      -p $APP_PORT:8501 \
                      $IMAGE_REPO:$IMAGE_TAG
                '''
            }
        }

        stage('Health Check') {
            steps {
                sh '''
                    sleep 15
                    curl -f http://localhost:$APP_PORT/_stcore/health
                    echo ""
                    echo "Streamlit app is healthy and running."
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded — deployed ${IMAGE_REPO}:${IMAGE_TAG} on port ${APP_PORT}"
        }
        failure {
            echo 'Pipeline failed — check the stage logs above.'
            sh 'docker logs $CONTAINER_NAME --tail 50 || true'
        }
        always {
            sh 'docker logout || true'
        }
    }
}
