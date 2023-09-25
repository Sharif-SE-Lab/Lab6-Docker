# Lab 6: Utilizing Docker
## Problem Definition
We intend to learn how to use Docker as an Orchestration and Deployment tool. As a result, We define a set of real-world requirements and design a software system that realizes those requirements.

# Requirements Analysis
A clinic has requested **a machine learning model for diabetes patient data**. The system is expected to **record the results** of multiple patients' tests on a daily basis, and the system should continue to **learn from patient data over time**.

Additionally, the system needs to have **the capability to predict the test results** for patients so that the clinic's physicians can sometimes use the machine learning model's predictions before the actual test results are available.

All communications with this system should be realized through a **RESTful API**, and there is no need to design and create a Front-End interface for this system, as the Front-End subsystem is going to be outsourced. The system should have a reasonable **load balancer** to ensure accessibility is not compromised during peak patient visit times.

# Software System Design
We leverage a Top-Down view to describe the designed system. Consequently, we first propose a top-level deployment diagram. Furthermore, we provide component diagrams for each of the deployment diagram nodes.

## Deployment Diagram
Initially, we develop a deployable node for the RESTful API that was requested. We hide this node behind an NGINX service, a well-known load-balancing web server. Additionally, we utilize a PostgreSQL database to preserve the patient data in a persistent storage.

![deployment diagram part 1](<./resources/deployment_diagram_pt1.png>)

As it is figured above, two instances of the `SELabAPI` gets deployed and the load balancer (NGINX) distributes the requests across the deployed instances with a `least_conn` policy; according to this policy, each request will be assigned to the node that is currently handling the minimum number of requests.

Furthermore, a simple regression model is required for learning and predicting the clinic data.

![deployment diagram part 2](./resources/deployment_diagram_pt2.png)

This model is implemented in Python. Other services can require its service through a RabbitMQ instance. As a result, the `API` node will use its service with minimum possible coupling.

**The overall deployment diagram of the whole system is represented bellow.**

![deployment diagram](./resources/deployment_diagram.jpg)

Two subsystem components exist in the above diagram (`API` and `Model`) that will be further discussed in the component diagram section; the word subsystem implies the top-level components that will be executed on each node.

## Component Diagram
For having a more refined view of the software system, we further study the component-level design of this system.

### `Model`
This component will be responsible to train the machine-learning models. It can also load previously trained models from disk if they exist or create new ones otherwise.
It consists of the following subcomponents:
- Regression: contains the regression model that trains and predicts on the diabetes patient data.
- DataLoader: this component is responsible for loading the data on the ML model.
- FeatureCleanser: cleanse the features of the dataset.
- Services: the wrapper that integrates all other subcomponents and provides the external service of the `Model` component.

`Model` provides two Facades (interfaces) for requiring its services, `ModelFacade` and `ModelConfigurationFacade`. The first one is further used to train and predict the new data that comes from the `API`. The second one can be used for changing the hyper-parameters of the model.
This Facades are implemented by utilizing the [nameko](https://www.nameko.io/) framework and using their methods (class operations) is further realized through RPC over AMQP. Moreover, queues of RabbitMQ are used due to storing the RPC messages.

![component diagram part 1](./resources/component_diagram_pt1.png)

Moreover, a RESTful API is implemented in Django Rest Framework. the django server is further hidden behind a uWSGI gateway. uWSGI stands between th `NGINX` and `API` components. The `API` component also requires the facades provided by `Model` for giving the services to the external clients.

![component diagram part 2](./resources/component_diagram_pt2.png)

**The following figure illustrates the component diagram of the whole system.**

![component diagram](./resources/component_diagram.jpg)

## Building and Installation
The project consists of three docker-compose files. the first configuration that we launch is the one located in the root of the project (next to this README file). It builds the Postgres and RabbitMQ components of the system alongside the docker network used across different services.
```bash
$ docker-compose up --build --detach
```
![first deploy](./resources/deploy/first_docker_compose.png)

We also use the following command instead of `docker ps` to get a brief status of the docker containers that we only care about in this project.
```bash
$ docker ps --format "table {{.Image}}  {{.Ports}}  {{.Names}}" | grep "selab"
``` 
The second configuration that we launch is located in the API component folder (directory name: diabetes_api). This configuration deploys the components of API nodes and NGINX. According to the provided configuration for NGINX, this web proxy serves on port 8088 of localhost and balances requests between two instances of the deployed API node.

![second deploy](./resources/deploy/second_docker_compose.png)

And finally, we deploy the configuration related to the Model node. The processes resulting from deployment actually serve the requests that come to the facades of the component under this node (to understand these facades, refer to the sub-section on the component diagram).
These facades, using the nameko package, read incoming requests from RabbitMQ queues and place the responses on other predefined RabbitMQ queues.

![third deploy](./resources/deploy/third_docker_compose.png)

Finally, the commands docker ps and docker image ls indicate the correctness of the container execution.

![docker status](./resources/deploy/docker_status.png)

## Endpoints
#### `POST` /api/data/patient/
for storing the patients data.

![post patient](./resources/api/post_patient.png)

#### `GET | PATCH | DELETE` /api/data/patient/<int:pk>
for updating or deleting a specific patient using their ID.

![get patient](./resources/api/get_patient.png)

#### `PATCH` /api/data/patient/<int:pk>/predict/
for predicting the result of a patient's test using the regression model.

![predict patient](./resources/api/predict_patient.png)

### `PATCH` /api/data/model/
for changing the model hyper-parameters.

![patch model](./resources/api/patch_model.png)

# Questions
Continuing with the answers to three questions:

### 1. Which UML diagrams have you used for modeling your MicroService architecture?
According to agile design principles, we initially created a Deployment diagram to represent the project's infrastructure architecture, followed by a Component diagram to illustrate its software architecture. Implementation based on these models was then initiated.

### 2. What is the relationship between Domain-driven Design (DDD) and MicroService architecture? Please explain in two to three sentences.
Domain-driven Design (DDD) is a software development approach that focuses on building a domain model rich in understanding the domain's processes and rules. By using DDD, one can break down the problem space into understandable components, even from the customer's perspective, and assign Microservices to address these smaller problem components.

### 3. Is Docker Compose an Orchestration tool? Please explain in two to three sentences.
Orchestration tools in software help us control, scale, and monitor our software components simultaneously and without the need for manual intervention on their environments. All of these activities can be achieved using Docker Compose. Docker Compose allows multiple Docker services to be defined in configuration files, and it can be used via the CLI to orchestrate their deployment, much like what we did in this experiment.
