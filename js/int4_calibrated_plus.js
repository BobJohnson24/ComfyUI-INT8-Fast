// Shows the calibrated_plus_preset dropdown only when selection == "calibrated+"
// on the INT4 W4A4 mixed converter node.
import { app } from "../../scripts/app.js";

const NODE_CLASSES = new Set(["INT4W4A4MixedSave"]);
const SELECTION_WIDGET = "selection";
const PRESET_WIDGET = "calibrated_plus_preset";

function setVisible(node, widget, show) {
    if (!widget) return;
    if (widget._int8fastOrigType === undefined) {
        widget._int8fastOrigType = widget.type;
        widget._int8fastOrigComputeSize = widget.computeSize;
    }
    if (show) {
        widget.type = widget._int8fastOrigType;
        widget.computeSize = widget._int8fastOrigComputeSize;
    } else {
        widget.type = "int8fastHidden";
        widget.computeSize = () => [0, -4]; // collapse the row (-4 swallows widget spacing)
    }
    widget.hidden = !show;
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "INT8Fast.CalibratedPlusPreset",
    nodeCreated(node) {
        if (!NODE_CLASSES.has(node.comfyClass)) return;
        const sel = node.widgets?.find((w) => w.name === SELECTION_WIDGET);
        const preset = node.widgets?.find((w) => w.name === PRESET_WIDGET);
        if (!sel || !preset) return;

        const update = () => setVisible(node, preset, sel.value === "calibrated+");

        const origCallback = sel.callback;
        sel.callback = function (...args) {
            const r = origCallback?.apply(this, args);
            update();
            return r;
        };

        // Apply on creation and again after workflow deserialization restores
        // widget values.
        requestAnimationFrame(update);
        const origConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const r = origConfigure?.apply(this, args);
            requestAnimationFrame(update);
            return r;
        };
    },
});
