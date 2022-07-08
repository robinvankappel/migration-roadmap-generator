### Introduction
Often a software company runs into issues with the used technology, where it is out of date, not supported anymore or not suitable for the current business needs. In those cases often a migration strategy is required where the current application with multiple features need to be replaced. Since this is often a large expensive project it is of great importance an appropriate roadmap is selected where the most used and/or valuable features are migrated first. This program provides a solution for such a use case, where it is assumed that for each client it can be determined whether it actively uses a certain feature or not. 

### Program description
Brute force method to generate a roadmap which facilitates migration in an optimal manner (either quickest migration of subscription quantity or MRR).

As input it uses a xls file with feature usage per client and client characteristics.

Pre-optimisation multiple optimisation segments can be set on which the optimisation is performed, defined by a range of  subscription_quantity (respresenting users for example). For each of these segments one solution is found.

For the most optimal roadmap a (post-)analysis is performed. Per segment the number of clients and associated revenue (MRR) is calculated. The graphical results are saved as images.

