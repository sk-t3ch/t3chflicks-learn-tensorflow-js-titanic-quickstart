<template>
  <v-app>
    <v-app-bar app color="indigo" dark>
      <v-avatar size="24">
          <v-img
            src="https://www.google.com/s2/favicons?domain=https://www.tensorflow.org/js"
            style="filter: blur(0.5px);"
          />
       </v-avatar>
      <div class="d-flex align-center subtitle-2 ml-2">Tensorflow JS - 
        <i>Titanic Quickstart</i>
      </div>
      <v-spacer></v-spacer>
    </v-app-bar>

    <v-main>
      <v-container>
        <HelloWorld />
      </v-container>
    </v-main>
  </v-app>
</template>

<script>
import HelloWorld from "./components/HelloWorld";

export default {
  name: "App",
  components: {
    HelloWorld,
  },
};
</script>
