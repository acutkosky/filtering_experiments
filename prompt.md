I've saved a paper in datafiltering.pdf. The latex source is available in datafiltering_latex/.
The entry point for the latex source is main.tex. If you can understand the paper from just the pdf that's fine. BUt if you read the latex, you should ignore any
files that are not actually used (they may be obselete), and also anything that is commented out.

I want to add some synthetic experiments to this paper.
Can you design some experiments that generate simulated data and run the algorithm to see the results?
You can use linear svm or logistic regression in place of exact solving of the linear classification problem since that is computationally difficult.

You should generate a README describing your experiments, and some plots that illustrate them well that I might be able to include in the paper.
Don't actually modify the paper though.

you should write code in python and use uv to manage dependencies.

Think critically about which experiments you are running. Make sure that they can run easily on a macbook in at most an hour.

