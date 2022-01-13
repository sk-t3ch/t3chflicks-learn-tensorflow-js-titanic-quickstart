<template>
  <div class="hello">
    <v-card color="blue" dark>
      <v-card-title class="indigo">
        Dataset
        <v-row>
          <v-spacer />
          <v-btn class="mx-2" small @click="downloadDataset"
            >Refresh <v-icon>mdi-refresh</v-icon></v-btn
          >
          <v-btn class="mx-2" small @click="dropNa"
            >drop NA <v-icon>mdi-delete</v-icon></v-btn
          >
          <v-btn class="mx-2" small @click="describeDataset"
            >Describe <v-icon>mdi-brightness-percent</v-icon></v-btn
          >
          <v-btn class="mx-2" small @click="printDataset"
            >Print <v-icon>mdi-eye</v-icon></v-btn
          >
          <v-btn class="mx-2" small @click="showViz"
            >TF Viz<v-icon>mdi-page-layout-sidebar-right</v-icon></v-btn
          >
          
        </v-row>
      </v-card-title>
      <v-card-text v-if="dataset" class="pt-5">
        <div id="columnns" style="overflow: scroll; max-height: 200px">
          <vue-json-editor
            v-model="editorText"
            mode="code"
            :modes="['code']"
            :show-btns="false"
            :exapnded-on-start="true"
          />
        </div>
      </v-card-text>
    </v-card>
    <div class="my-2">
      <hr />
    </div>
    <v-card color="blue" dark v-if="dataset">
      <v-card-title class="indigo">
        Survival Classifier <v-spacer />
      </v-card-title>
      <v-card-text class="pt-5">
        <div>
          <ul>
          <v-btn small class="ma-1" @click="preprocessDataset">1. Preprocess Dataset</v-btn>
          <v-btn small class="ma-1" @click="createModel">      2. Create Model   </v-btn>
          <v-btn small class="ma-1" @click="train">            3. Train          </v-btn>
          <v-btn small class="ma-1" @click="makePredictions">  4. Predict        </v-btn>
          </ul>
        </div>
      </v-card-text>
    </v-card>
    <div class="my-2">
      <hr />
    </div>
    <v-card color="blue" dark v-if="dataset">
      <v-card-title class="indigo">
        Plot 
        <v-row class="pr-2">
          <v-spacer />
          <v-select dense rounded outlined class="pt-3" style="width: 120px" :items="dataset.columns" v-model="factorA"></v-select>
          <v-select dense rounded outlined class="pt-3" style="width: 120px" :items="dataset.columns" v-model="factorB"></v-select>
          <v-btn @click="factorPlot" small class="mt-3 ml-2">Factor</v-btn>
          <v-btn @click="violinPlot" small class="mt-3 ml-2">Violin</v-btn>
        </v-row>
      </v-card-title>
      <v-card-text class="pt-5">
        <div id="plot_div"></div>
      </v-card-text>
    </v-card>
  </div>
</template>

<script>
const dfd = require("danfojs");
const tf = dfd.tf; //Reference to the exported tensorflowjs library
import * as tfvis from "@tensorflow/tfjs-vis";
import vueJsonEditor from 'vue-json-editor'

console.log(tf);
export default {
  name: "HelloWorld",
  props: {
    msg: String,
  },
  components: {
    vueJsonEditor
  },
  data() {
    return {
      chosenDataset: null,
      mostRead: null,
      df: null,
      dataset: null,
      datasetDescribe: null,
      editorText: '',
      showDataset: false,
      factorA: null,
      factorB: null,
      model: null,
      isDownloading: false,
      xTrain: null,
      yTrain: null
    };
  },
  created() {
    this.chosenDataset = {
          title: 'Titanic',
          url: "https://raw.githubusercontent.com/sk-t3ch/test/main/titanic.csv"
    }
    this.downloadDataset()
  },
  methods: {
    showViz() {
      tfvis.visor().open()
    },
    printDataset() { 
      this.editorText = this.dataset.head()
    },
    dropNa() {
      this.dataset.dropna(0, { inplace: true })
    },
    describeDataset() { 
      // this.datasetDescribe = null;
      // let values = ["Apples", df["Count"].mean()]
      // let df_filled = df.fillna(values, { columns: ["Name", "Count"] })
      const describe = this.dataset.describe();
      
      console.log(describe)
      this.datasetDescribe =  describe.columns.reduce( (acc, column) => {
        acc[column] = {}
        describe[column].$index.map( (stat, idx) => {
          acc[column][stat] = describe[column].$data[idx]
        })
        return acc
      }, {})
      this.editorText = this.datasetDescribe
      console.log(this.datasetDescribe)
    },
    async downloadDataset() {
      this.isDownloading = true
      let df = await dfd.read_csv(this.chosenDataset.url);
      console.log(df);
      this.dataset = df;
      this.printDataset();
      this.factorA = df.columns[0]
      this.factorB = df.columns[1]
      this.isDownloading = false
    },
    async preprocessDataset() {
      this.editorText = 'preprocessing data'
      let title = this.dataset["Name"].apply((x) => {
        return x.split(".")[0];
      }).values;
      this.dataset.addColumn({ column: "Name", values: title, inplace: true });
      this.editorText += '... finished feature engineering'

      let encoder = new dfd.LabelEncoder();
      let cols = ["Sex", "Name"];
      cols.forEach((col) => {
        encoder.fit(this.dataset[col]);
        var enc_val = encoder.transform(this.dataset[col]);
        this.dataset.addColumn({ column: col, values: enc_val, inplace: true });
      });
      this.editorText += '... finished label encode'

      let Xtrain = this.dataset.iloc({ columns: [`1:`] });
      let ytrain = this.dataset["Survived"];
      
      let scaler = new dfd.MinMaxScaler();
      scaler.fit(Xtrain);
      Xtrain = scaler.transform(Xtrain);
      this.editorText += '... finished minMaxScaling'

      this.editorText += `${JSON.stringify(Xtrain.$data)} ${JSON.stringify(ytrain.$data)}`
      this.Xtrain = Xtrain
      this.ytrain = ytrain

      await tfvis.show.valuesDistribution({ name: "True Distribution", tab: "Model Inspection" }, ytrain.tensor);
      const surface = { name: 'Table', tab: 'Charts' };
      // TODO: fix table to show train dataset
      tfvis.render.table(surface, { headers: this.dataset.columns.slice(1, this.dataset.columns.length), values: this.XtrainT });
      tfvis.visor().show()

      // // await tfvis.show.valuesDistribution(
      //   { name: "Result Distribution", tab: "Model Inspection" },
      //   tf.tensor1d(boolPreds)
      // );
      // console.log('finished tfv')
    },
    createModel() {
      const model = tf.sequential();
      model.add(
        tf.layers.dense({
          inputShape: [7],
          units: 124,
          activation: "relu",
          kernelInitializer: "leCunNormal",
        })
      );
      model.add(tf.layers.dense({ units: 64, activation: "relu" }));
      model.add(tf.layers.dense({ units: 32, activation: "relu" }));
      model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
      model.summary();
      this.model = model
      this.editorText  = 'finished createModel'
      tfvis.show.modelSummary({ name: "Model Summary" }, this.model);
      tfvis.visor().open()
    },
    async train() {
      const XtrainT = this.Xtrain.tensor;
      const ytrainT = this.ytrain.tensor;

      this.model.compile({
        optimizer: "rmsprop",
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
      });

      console.log("Training started....");
      tfvis.visor().open()
      await this.model.fit(XtrainT, ytrainT, {
        batchSize: 32,
        epochs: 15,
        validationSplit: 0.2,
        callbacks: tfvis.show.fitCallbacks(
          { name: "Training Performance" },
          ["loss", "mse"],
          { height: 200, callbacks: ["onEpochEnd"] }
        ),
      });
      console.log('finished train')
    },
    async makePredictions() {
      const XtrainT = this.Xtrain.tensor;
      const ytrainT = this.ytrain.tensor;
      const preds = this.model.predict(XtrainT);
      const boolPreds = preds.dataSync().map((x) => Math.round(x)); // 0.5 threshold

      const perClassAccuracy = await tfvis.metrics.perClassAccuracy(
        ytrainT,
        tf.tensor1d(boolPreds)
      );
      
      const labels = ["Dead", "Alive"];
      await tfvis.show.perClassAccuracy(
        { name: "Result Distribution", tab: "Model Inspection" },
        perClassAccuracy,
        labels
      );

      const confusionMatrix = await tfvis.metrics.confusionMatrix(
        ytrainT,
        tf.tensor1d(boolPreds)
      );

      await tfvis.render.confusionMatrix(
        { name: "Confusion Matrix with Excluded Diagonal", tab: "Charts" },
        {
          values: confusionMatrix,
          tickLabels: labels,
        }
      );
      tfvis.visor().open()
    },
    violinPlot() {
      this.dataset.plot("plot_div").violin({ x: this.factorA, y: this.factorB });
    },
    factorPlot() {
      const uniqueValuesFactorA = this.dataset[this.factorA].unique().$data;
      const uniqueValuesFactorB = this.dataset[this.factorB].unique().$data;
      const resultsArray = new Array(uniqueValuesFactorB.length)
        .fill(0)
        .map(() => new Array());

      uniqueValuesFactorA.map((uniqueA) => {
        const factorBWithUniqueA = this.dataset.iloc({
          rows: this.dataset[this.factorA].eq(uniqueA),
        })[this.factorB];
        const factorBValueCountsForUniqueA =
          factorBWithUniqueA.value_counts().$data;
        factorBValueCountsForUniqueA.map((g, idx) => {
          resultsArray[idx].push(g);
        });
      });

      const mf = new dfd.DataFrame(resultsArray, {
        columns: uniqueValuesFactorA,
      });
      mf.plot("plot_div").bar({
        layout: {
          title: {
            x: 20,
            text: `${this.factorA} factor ${this.factorB} Plot`,
          },
          xaxis: {
            title: this.factorB,
          },
          yaxis: {
            title: `${this.factorA} count`,
          },
        },
      });
    }
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style >
div.jsoneditor-menu {
  display: none;
}
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
