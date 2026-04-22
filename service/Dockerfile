ARG MAVEN_VERSION=3.9.6
ARG JDK_VERSION=21

FROM maven:${MAVEN_VERSION}-eclipse-temurin-${JDK_VERSION} AS build
WORKDIR /src
COPY pom.xml .
RUN mvn -B -ntp dependency:go-offline
COPY src ./src
RUN mvn -B -ntp -DskipTests package

FROM eclipse-temurin:${JDK_VERSION}-jre AS runtime
WORKDIR /app
RUN groupadd -g 10001 ga && useradd -m -u 10001 -g ga ga
COPY --from=build /src/target/geoalpha-service-*.jar /app/app.jar
USER ga
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/actuator/health | grep -q UP || exit 1
ENV JAVA_OPTS="-XX:+UseG1GC -XX:MaxRAMPercentage=70"
ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar /app/app.jar"]
